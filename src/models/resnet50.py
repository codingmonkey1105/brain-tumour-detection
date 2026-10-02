"""ResNet50 transfer-learning model (frozen ImageNet base).

Uses the same classification head as VGG16 so the comparison is fair.
Note: inputs are rescaled to [0, 1] by the shared generator pipeline.
"""
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam


def build_resnet50(input_shape=(224, 224, 3), lr=1e-4, dropout=0.5, dense_units=128, unfreeze_last_block=False):
    base = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    base.trainable = False
    if unfreeze_last_block:  # planned fine-tuning step: last residual stage (conv5_*)
        base.trainable = True
        for layer in base.layers:
            layer.trainable = layer.name.startswith("conv5")
    model = models.Sequential(
        [
            base,
            layers.GlobalAveragePooling2D(),
            layers.Dense(dense_units, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="resnet50_binary",
    )
    model.compile(optimizer=Adam(lr), loss="binary_crossentropy", metrics=["accuracy"])
    return model
