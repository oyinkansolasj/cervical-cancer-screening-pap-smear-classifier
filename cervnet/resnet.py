"""A compact pre-activation ResNet for small medical images, written with tf.keras."""
from tensorflow.keras import layers, models, regularizers


def _residual_block(x, filters, stride, reg, project):
    """Bottleneck residual block (BN -> ReLU -> Conv) x3 with an identity/projection shortcut."""
    shortcut = x
    pre = layers.BatchNormalization()(x)
    pre = layers.Activation("relu")(pre)

    y = layers.Conv2D(filters // 4, 1, use_bias=False,
                      kernel_regularizer=regularizers.l2(reg))(pre)
    y = layers.BatchNormalization()(y)
    y = layers.Activation("relu")(y)
    y = layers.Conv2D(filters // 4, 3, strides=stride, padding="same", use_bias=False,
                      kernel_regularizer=regularizers.l2(reg))(y)
    y = layers.BatchNormalization()(y)
    y = layers.Activation("relu")(y)
    y = layers.Conv2D(filters, 1, use_bias=False,
                      kernel_regularizer=regularizers.l2(reg))(y)

    if project:
        shortcut = layers.Conv2D(filters, 1, strides=stride, use_bias=False,
                                 kernel_regularizer=regularizers.l2(reg))(pre)
    return layers.Add()([y, shortcut])


def build_resnet(input_shape, num_classes, stages=(3, 4, 6), filters=(64, 128, 256, 512),
                 reg=5e-4, augmentation=None):
    """Build the network.

    stages  -- number of residual blocks in each stage
    filters -- filters for the stem conv followed by one entry per stage
    """
    inputs = layers.Input(shape=input_shape)
    x = augmentation(inputs) if augmentation is not None else inputs
    x = layers.Rescaling(1.0 / 255)(x)

    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters[0], 3, padding="same", use_bias=False,
                      kernel_regularizer=regularizers.l2(reg))(x)

    for i, num_blocks in enumerate(stages):
        stride = 1 if i == 0 else 2
        x = _residual_block(x, filters[i + 1], stride, reg, project=True)
        for _ in range(num_blocks - 1):
            x = _residual_block(x, filters[i + 1], 1, reg, project=False)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax",
                           kernel_regularizer=regularizers.l2(reg))(x)
    return models.Model(inputs, outputs, name="cervnet_resnet")
