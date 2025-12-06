# realtime_demo.py
import cv2
from utils_mm import inference
from tensorflow.keras.models import load_model

model = load_model('multimodal.h5')
cap   = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break
    cls, conf = inference(model, frame)
    if cls:
        cv2.putText(frame, f"{cls} {conf:.2f}", (10,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
    cv2.imshow("MM-SLB", frame)
    if cv2.waitKey(1) & 0xFF == 27: break   # ESC keluar
cap.release(); cv2.destroyAllWindows()