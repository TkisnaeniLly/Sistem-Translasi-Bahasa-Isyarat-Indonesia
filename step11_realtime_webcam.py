import cv2
import numpy as np
from tensorflow.keras.models import load_model

# =====================
# KONFIGURASI
# =====================
MODEL_PATH = "model_bisindo_mobilenet.h5"
IMG_SIZE = 224
CLASS_NAMES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
CONF_THRESHOLD = 0.60  # 60%

# =====================
# LOAD MODEL
# =====================
model = load_model(MODEL_PATH)

# =====================
# BUKA WEBCAM
# =====================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Webcam tidak bisa dibuka")
    exit()

print("Webcam aktif | Tekan Q untuk keluar")

# =====================
# LOOP REAL-TIME
# =====================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # =====================
    # ROI (AREA TANGAN)
    # =====================
    x1, y1, x2, y2 = 100, 100, 400, 400
    roi = frame[y1:y2, x1:x2]

    # PREPROCESS
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # PREDIKSI
    preds = model.predict(img, verbose=0)[0]
    pred_idx = np.argmax(preds)
    confidence = preds[pred_idx]

    # =====================
    # TAMPILKAN HASIL
    # =====================
    if confidence >= CONF_THRESHOLD:
        label = f"{CLASS_NAMES[pred_idx]} ({confidence*100:.2f}%)"
        color = (0, 255, 0)
    else:
        label = "Tidak dikenali"
        color = (0, 0, 255)

    # Gambar kotak ROI
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame, label, (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
    )

    cv2.imshow("BISINDO Real-Time Sign to Text", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =====================
# CLEAN UP
# =====================
cap.release()
cv2.destroyAllWindows()
