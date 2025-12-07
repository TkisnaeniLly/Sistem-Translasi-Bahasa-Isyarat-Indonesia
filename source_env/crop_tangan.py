import cv2
import mediapipe as mp
import os

input_folder = 'BISINDO'
output_folder = 'dataset_cropped'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

for letter in os.listdir(input_folder):
    letter_folder = os.path.join(input_folder, letter)
    save_folder = os.path.join(output_folder, letter)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for img_name in os.listdir(letter_folder):
        img_path = os.path.join(letter_folder, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, _ = image.shape
                x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
                y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
                x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
                y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)

                pad = 10
                x_min = max(x_min - pad, 0)
                y_min = max(y_min - pad, 0)
                x_max = min(x_max + pad, w)
                y_max = min(y_max + pad, h)

                crop_img = image[y_min:y_max, x_min:x_max]

                save_path = os.path.join(save_folder, img_name)
                cv2.imwrite(save_path, crop_img)
        else:
            print(f"Tangan tidak terdeteksi di {img_path}")
