"""VGG16 transfer-learning model (frozen ImageNet base)."""
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam


def build_vgg16(input_shape=(224, 224, 3), lr=1e-4, dropout=0.5, dense_units=128, unfreeze_last_block=False):
    base = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    base.trainable = False
    if unfreeze_last_block:  # planned fine-tuning step: last conv block only
        base.trainable = True
        for layer in base.layers:
            layer.trainable = layer.name.startswith("block5")
    model = models.Sequential(
        [
            base,
            layers.GlobalAveragePooling2D(),
            layers.Dense(dense_units, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="vgg16_binary",
    )
    model.compile(optimizer=Adam(lr), loss="binary_crossentropy", metrics=["accuracy"])
    return model
