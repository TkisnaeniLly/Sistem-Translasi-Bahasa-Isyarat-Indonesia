import os
import cv2
import numpy as np
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

total = 0
correct = 0

print("\n=== HASIL PENGUJIAN SEMUA FOLDER ===\n")

# =====================
# LOOP SETIAP FOLDER
# =====================
for label in sorted(os.listdir(TEST_DIR)):
    folder_path = os.path.join(TEST_DIR, label)

    if not os.path.isdir(folder_path):
        continue

    print(f"\nFolder {label}:")
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
            pred_label = CLASS_NAMES[pred_idx]
            confidence = np.max(pred)

            total += 1
            if pred_label == label:
                correct += 1
                status = "✔"
            else:
                status = "✖"

            print(f"  {file:25s} → {pred_label} ({confidence*100:.2f}%) {status}")

# =====================
# AKURASI TOTAL
# =====================
accuracy = (correct / total) * 100 if total > 0 else 0

print("\n==============================")
print(f"Total data uji : {total}")
print(f"Benar         : {correct}")
print(f"Akurasi       : {accuracy:.2f}%")
print("==============================")
