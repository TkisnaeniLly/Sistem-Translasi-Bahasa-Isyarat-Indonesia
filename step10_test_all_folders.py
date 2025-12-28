import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# =====================
# KONFIGURASI
# =====================
MODEL_PATH = "model_bisindo_mobilenet.h5"
TEST_DIR = "test_images"
IMG_SIZE = 224
CLASS_NAMES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# =====================
# LOAD MODEL
# =====================
model = load_model(MODEL_PATH)

# =====================
# VARIABEL HASIL
# =====================
accuracy_per_class = {}

# =====================
# LOOP SETIAP FOLDER HURUF
# =====================
for idx, label in enumerate(CLASS_NAMES):
    folder_path = os.path.join(TEST_DIR, label)

    if not os.path.exists(folder_path):
        continue

    total = 0
    correct = 0

    for file in os.listdir(folder_path):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(folder_path, file)

            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0
            img = np.expand_dims(img, axis=0)

            pred = model.predict(img, verbose=0)
            pred_idx = np.argmax(pred)

            total += 1
            if pred_idx == idx:
                correct += 1

    if total > 0:
        accuracy = (correct / total) * 100
        accuracy_per_class[label] = accuracy
    else:
        accuracy_per_class[label] = 0.0

# =====================
# TAMPILKAN HASIL TEKS
# =====================
print("\nAkurasi per Huruf:")
for k, v in accuracy_per_class.items():
    print(f"{k} : {v:.2f}%")

# =====================
# PLOT GRAFIK
# =====================
labels = list(accuracy_per_class.keys())
values = list(accuracy_per_class.values())

plt.figure(figsize=(14, 6))
bars = plt.bar(labels, values)

plt.xlabel("Huruf")
plt.ylabel("Akurasi (%)")
plt.title("Akurasi Model BISINDO per Huruf (MobileNetV2)")
plt.ylim(0, 100)

# TULIS ANGKA DI ATAS BAR (2 DESIMAL)
for bar, value in zip(bars, values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.2f}%",
        ha='center',
        va='bottom',
        fontsize=9
    )

plt.tight_layout()
plt.savefig("akurasi_per_huruf.png", dpi=300)
plt.show()
