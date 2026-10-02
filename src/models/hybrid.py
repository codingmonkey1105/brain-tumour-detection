"""Hybrid VGG16-ResNet50 model (IN PROGRESS: training/evaluation not yet completed).

Pooled features from both frozen backbones are concatenated, then dense + dropout
and a sigmoid output layer.
"""
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.optimizers import Adam


def build_hybrid(input_shape=(224, 224, 3), lr=1e-4, dropout=0.5, dense_units=256):
    inputs = layers.Input(shape=input_shape, name="img")

    vgg = VGG16(weights="imagenet", include_top=False, input_tensor=inputs)
    vgg.trainable = False
    vgg_feat = layers.GlobalAveragePooling2D(name="vgg_gap")(vgg.output)

    # Both backbones read the same input tensor.
    resnet = ResNet50(weights="imagenet", include_top=False, input_tensor=inputs)
    resnet.trainable = False
    res_feat = layers.GlobalAveragePooling2D(name="resnet_gap")(resnet.output)

    x = layers.Concatenate(name="fusion")([vgg_feat, res_feat])
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = Model(inputs, out, name="hybrid_vgg16_resnet50")
    model.compile(optimizer=Adam(lr), loss="binary_crossentropy", metrics=["accuracy"])
    return model
