# utils_mm.py
import numpy as np, cv2, mediapipe as mp, joblib, tensorflow as tf

IMG_SIZE   = 128
SENSOR_LEN = 8                     # <-- sesuaikan dengan rf_model
CLASS      = ['A','B','C','D','E','F','G','H','I','J']   # contoh

# ---------- 1. load Random-Forest untuk “sensor fake” ----------
rf = joblib.load('rf_model.pkl')

# ---------- 2. Mediapipe ----------
mp_hands = mp.solutions.hands
hands    = mp_hands.Hands(static_image_mode=False,
                          max_num_hands=1,
                          min_detection_confidence=0.7)
mp_draw  = mp.solutions.drawing_utils

def crop_hand(img, landmarks):
    h, w = img.shape[:2]
    xs = [lm.x*w for lm in landmarks.landmark]
    ys = [lm.y*h for lm in landmarks.landmark]
    x1, y1 = int(min(xs))-20, int(min(ys))-20
    x2, y2 = int(max(xs))+20, int(max(ys))+20
    x1, y1 = max(0,x1), max(0,y1)
    x2, y2 = min(w,x2), min(h,y2)
    return img[y1:y2, x1:x2]

# ---------- 3. sensor “palsu” ----------
def fake_sensor_from_image(img, landmarks):
    """
    Ekstrak fitur sederhana dari landmark -> infer sensor
    (versi cepat, cukup untuk demo)
    """
    h, w = img.shape[:2]
    xs = [lm.x*w for lm in landmarks.landmark]
    ys = [lm.y*h for lm in landmarks.landmark]
    # 8 fitur: 4 sudut + 4 jarak
    feature = np.array([
        *np.std([xs,ys], axis=1),          # std x, std y
        np.ptp(xs), np.ptp(ys),            # range
        np.mean(xs), np.mean(ys),
        (xs[4]-xs[8]), (ys[4]-ys[8])       # jarak jempol-jari telunjuk
    ])[:SENSOR_LEN]
    feature = feature.reshape(1,-1)
    sensor  = rf.predict(feature)[0]      # kelas RF 0..N-1
    # one-hot versi tipis -> distribusi sensor
    prob = np.eye(len(CLASS))[sensor] + np.random.randn(len(CLASS))*0.05
    prob = np.clip(prob,0,1)
    return prob/np.sum(prob)

# ---------- 4. preprocessing image ----------
def preprocess_img(img_bgr):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype('float32') / 255.
    return img

# ---------- 5. inference ----------
def inference(model, frame):
    """
    frame : BGR dari cv2.VideoCapture
    return: (predicted_class, confidence)
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)
    if not res.multi_hand_landmarks:
        return None, 0.0

    hand = res.multi_hand_landmarks[0]
    roi  = crop_hand(frame, hand)
    img  = preprocess_img(roi)
    img  = np.expand_dims(img,0)          # (1,128,128,3)

    # sensor “palsu”
    sensor_prob = fake_sensor_from_image(frame, hand)
    sensor_prob = np.expand_dims(sensor_prob,0)  # (1,10) misal

    pred = model.predict([sensor_prob, img])[0]
    cls  = CLASS[np.argmax(pred)]
    conf = float(np.max(pred))
    return cls, conf