from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Dropout, GlobalAveragePooling2D,
    GaussianNoise
)
from tensorflow.keras.optimizers import Adam

IMG_SIZE = 64
NUM_CLASSES = 26

# === 1. BASE MODEL ===
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False  # transfer learning aman dulu

# === 2. HEAD MODEL ===
x = base_model.output

# ⬇️ Gaussian Noise HARUS DI SINI (SETELAH base_model ADA)
x = GaussianNoise(0.05)(x)

x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(NUM_CLASSES, activation="softmax")(x)

# === 3. FINAL MODEL ===
model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=0.0003),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Simpan model struktur (opsional)
model.save("model_bisindo_mobilenet.h5")
