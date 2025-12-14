import os
import cv2
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend untuk plotting
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow as tf

# Disable GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')

print("=" * 70)
print("BISINDO SIGN LANGUAGE RECOGNITION - INFERENCE")
print("=" * 70)

# ============================================================================
# LOAD MODEL DAN CLASS NAMES
# ============================================================================
print("\nLoading model...")
try:
    model = load_model('best_model.h5')
    print("✓ Model loaded successfully from 'best_model.h5'")
except:
    print("✗ Failed to load 'best_model.h5', trying 'final_model.h5'...")
    model = load_model('final_model.h5')
    print("✓ Model loaded successfully from 'final_model.h5'")

print("\nLoading class names...")
with open('class_names.pkl', 'rb') as f:
    class_names = pickle.load(f)
print(f"✓ Loaded {len(class_names)} classes: {class_names}")

IMG_SIZE = 64

# ============================================================================
# FUNGSI: PREDICT SINGLE IMAGE
# ============================================================================
def predict_image(image_path, model, class_names, img_size=64):
    """
    Predict gesture dari single image
    """
    # Load dan preprocess image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot read image {image_path}")
        return None
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (img_size, img_size))
    img_normalized = img_resized.astype('float32') / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)
    
    # Predict
    predictions = model.predict(img_input, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx] * 100
    predicted_class = class_names[predicted_class_idx]
    
    return predicted_class, confidence, predictions[0]

# ============================================================================
# FUNGSI: PREDICT FROM WEBCAM
# ============================================================================
def predict_from_webcam(model, class_names, img_size=64):
    """
    Real-time prediction dari webcam
    """
    print("\n" + "=" * 70)
    print("STARTING WEBCAM...")
    print("=" * 70)
    print("Instructions:")
    print("  - Press 'q' to quit")
    print("  - Press 's' to save screenshot")
    print("  - Show hand gesture in the green box")
    print("=" * 70 + "\n")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return
    
    # Set resolusi webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    screenshot_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame")
            break
        
        # Flip frame untuk efek mirror
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        # ROI (Region of Interest) untuk hand gesture
        roi_size = 300
        roi_x = (w - roi_size) // 2
        roi_y = (h - roi_size) // 2
        
        # Extract ROI
        roi = frame[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size]
        
        # Preprocess ROI untuk prediction
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi_resized = cv2.resize(roi_rgb, (img_size, img_size))
        roi_normalized = roi_resized.astype('float32') / 255.0
        roi_input = np.expand_dims(roi_normalized, axis=0)
        
        # Predict
        predictions = model.predict(roi_input, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx] * 100
        predicted_class = class_names[predicted_class_idx]
        
        # Draw ROI box
        cv2.rectangle(frame, (roi_x, roi_y), 
                     (roi_x+roi_size, roi_y+roi_size), 
                     (0, 255, 0), 2)
        
        # Display prediction
        text = f"{predicted_class}: {confidence:.1f}%"
        cv2.putText(frame, text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display top 3 predictions
        top_3_idx = np.argsort(predictions[0])[-3:][::-1]
        y_offset = 70
        for idx in top_3_idx:
            text = f"{class_names[idx]}: {predictions[0][idx]*100:.1f}%"
            cv2.putText(frame, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 30
        
        # Instructions
        cv2.putText(frame, "Press 'q' to quit | 's' to save", 
                   (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 255, 255), 1)
        
        # Show frame
        cv2.imshow('BISINDO Sign Language Recognition', frame)
        
        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            screenshot_count += 1
            filename = f'screenshot_{screenshot_count}.jpg'
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nWebcam closed.")

# ============================================================================
# FUNGSI: BATCH PREDICTION
# ============================================================================
def predict_batch(test_folder, model, class_names, img_size=64):
    """
    Predict multiple images dari folder
    """
    print(f"\nPredicting images from folder: {test_folder}")
    
    if not os.path.exists(test_folder):
        print(f"Error: Folder {test_folder} not found")
        return
    
    images = [f for f in os.listdir(test_folder) 
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if len(images) == 0:
        print("No images found in folder")
        return
    
    print(f"Found {len(images)} images\n")
    
    results = []
    for img_name in images:
        img_path = os.path.join(test_folder, img_name)
        result = predict_image(img_path, model, class_names, img_size)
        
        if result:
            predicted_class, confidence, _ = result
            results.append({
                'image': img_name,
                'prediction': predicted_class,
                'confidence': confidence
            })
            print(f"{img_name:40s} -> {predicted_class:15s} ({confidence:.2f}%)")
    
    return results

# ============================================================================
# MENU INTERFACE
# ============================================================================
def main():
    print("\n" + "=" * 70)
    print("SELECT MODE:")
    print("=" * 70)
    print("1. Predict single image")
    print("2. Predict from folder (batch)")
    print("3. Real-time webcam prediction")
    print("4. Exit")
    print("=" * 70)
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == '1':
        image_path = input("Enter image path: ").strip()
        if os.path.exists(image_path):
            result = predict_image(image_path, model, class_names, IMG_SIZE)
            if result:
                predicted_class, confidence, all_predictions = result
                print("\n" + "=" * 70)
                print("PREDICTION RESULT:")
                print("=" * 70)
                print(f"Predicted Class: {predicted_class}")
                print(f"Confidence: {confidence:.2f}%")
                print("\nAll Class Probabilities:")
                for i, class_name in enumerate(class_names):
                    print(f"  {class_name:15s}: {all_predictions[i]*100:.2f}%")
                print("=" * 70)
        else:
            print(f"Error: Image not found at {image_path}")
    
    elif choice == '2':
        folder_path = input("Enter folder path: ").strip()
        predict_batch(folder_path, model, class_names, IMG_SIZE)
    
    elif choice == '3':
        predict_from_webcam(model, class_names, IMG_SIZE)
    
    elif choice == '4':
        print("\nExiting...")
    
    else:
        print("\nInvalid choice!")

# ============================================================================
# RUN
# ============================================================================
if __name__ == "__main__":
    main()