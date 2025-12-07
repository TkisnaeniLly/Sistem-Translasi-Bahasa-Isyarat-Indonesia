import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pytz

from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

# Konfigurasi TensorFlow untuk CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)

print("=" * 70)
print("BISINDO TRAINING - MEMORY EFFICIENT (DATA GENERATOR)")
print("=" * 70)

# Setup waktu
timeJKT = pytz.timezone('Asia/Jakarta') 
start_time = datetime.now(timeJKT).replace(microsecond=0)
print(f"Start Time: {start_time}\n")

# ============================================================================
# KONFIGURASI
# ============================================================================
DATA_DIR = r"d:\PCD\Project\dataset"
IMG_SIZE = 64
BATCH_SIZE = 128  # Larger batch untuk generator
EPOCHS = 50
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2

# ============================================================================
# CHECK DATASET
# ============================================================================
print("Checking dataset structure...")
if not os.path.exists(DATA_DIR):
    print(f"Error: {DATA_DIR} not found!")
    exit(1)

class_folders = sorted([f for f in os.listdir(DATA_DIR) 
                       if os.path.isdir(os.path.join(DATA_DIR, f))])

if len(class_folders) == 0:
    print(f"Error: No class folders found in {DATA_DIR}")
    exit(1)

print(f"Found {len(class_folders)} classes: {class_folders}")

# Count images per class
total_images = 0
for folder in class_folders:
    folder_path = os.path.join(DATA_DIR, folder)
    num_images = len([f for f in os.listdir(folder_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    total_images += num_images
    print(f"  {folder}: {num_images:,} images")

print(f"\nTotal images: {total_images:,}")
print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}\n")

# ============================================================================
# DATA GENERATORS (MEMORY EFFICIENT!)
# ============================================================================
print("Setting up data generators...")

# Data augmentation untuk training (optional, data sudah di-augment)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=VALIDATION_SPLIT
)

# Generator untuk training set
train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

# Generator untuk validation set
val_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=42
)

num_classes = len(train_generator.class_indices)
class_names = list(train_generator.class_indices.keys())

print(f"✓ Training samples: {train_generator.samples:,}")
print(f"✓ Validation samples: {val_generator.samples:,}")
print(f"✓ Number of classes: {num_classes}")
print(f"✓ Steps per epoch: {train_generator.samples // BATCH_SIZE}")
print(f"✓ Validation steps: {val_generator.samples // BATCH_SIZE}\n")

# ============================================================================
# BUILD CNN MODEL
# ============================================================================
print("Building CNN model...")

model = Sequential([
    # Block 1
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    # Block 2
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    # Block 3
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    # Fully Connected
    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile model
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel Summary:")
model.summary()

# ============================================================================
# CALLBACKS
# ============================================================================
callbacks = [
    ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]

# ============================================================================
# TRAINING
# ============================================================================
print("\n" + "=" * 70)
print("STARTING TRAINING...")
print("=" * 70 + "\n")

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=val_generator,
    validation_steps=val_generator.samples // BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# Save final model
model.save('final_model.h5')
print("\n✓ Final model saved to 'final_model.h5'")

# Save class names
with open('class_names.pkl', 'wb') as f:
    pickle.dump(class_names, f)
print("✓ Class names saved to 'class_names.pkl'")

# ============================================================================
# PLOT TRAINING HISTORY
# ============================================================================
print("\nGenerating training visualizations...")

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Accuracy
axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss
axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Training history saved to 'training_history.png'")

# ============================================================================
# EVALUATION ON VALIDATION SET
# ============================================================================
print("\n" + "=" * 70)
print("EVALUATING MODEL...")
print("=" * 70 + "\n")

# Reset validation generator
val_generator.reset()

# Get predictions
print("Generating predictions...")
y_pred = model.predict(val_generator, steps=val_generator.samples // BATCH_SIZE, verbose=1)
y_pred_classes = np.argmax(y_pred, axis=1)

# Get true labels
y_true = val_generator.classes[:len(y_pred_classes)]

# Metrics
val_loss, val_accuracy = model.evaluate(val_generator, steps=val_generator.samples // BATCH_SIZE, verbose=0)
print(f"\nValidation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")



# ---------- CLASSIFICATION REPORT (hanya yg ada di data) ----------
labels_present = sorted(set(y_true) | set(y_pred_classes))
target_names   = [class_names[i] for i in labels_present]

print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes,
                            labels=labels_present,
                            target_names=target_names,
                            digits=4))

# ---------- CONFUSION MATRIX (bisa tetap full 29×29 atau ikut labels_present) ----------
cm = confusion_matrix(y_true, y_pred_classes,
                      labels=labels_present)   # biar ukuran sama

plt.figure(figsize=(max(6, len(labels_present)*.5),
                    max(6, len(labels_present)*.5)))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names,
            yticklabels=target_names,
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
plt.ylabel('True Label'); plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Confusion matrix saved to 'confusion_matrix.png'")

# ============================================================================
# SUMMARY
# ============================================================================
end_time = datetime.now(timeJKT).replace(microsecond=0)
duration = end_time - start_time

print("\n" + "=" * 70)
print("TRAINING COMPLETED!")
print("=" * 70)
print(f"Start Time: {start_time}")
print(f"End Time: {end_time}")
print(f"Total Duration: {duration}")
print(f"\nFinal Results:")
print(f"  - Validation Accuracy: {val_accuracy*100:.2f}%")
print(f"  - Validation Loss: {val_loss:.4f}")
print(f"\nFiles Generated:")
print("  ✓ best_model.h5 - Best model during training")
print("  ✓ final_model.h5 - Final model")
print("  ✓ class_names.pkl - Class labels")
print("  ✓ training_history.png - Training curves")
print("  ✓ confusion_matrix.png - Confusion matrix")
print("=" * 70)

# Save training summary
with open('training_summary.txt', 'w') as f:
    f.write("BISINDO TRAINING SUMMARY\n")
    f.write("=" * 70 + "\n")
    f.write(f"Start Time: {start_time}\n")
    f.write(f"End Time: {end_time}\n")
    f.write(f"Duration: {duration}\n")
    f.write(f"\nDataset:\n")
    f.write(f"  - Total Images: {total_images:,}\n")
    f.write(f"  - Training: {train_generator.samples:,}\n")
    f.write(f"  - Validation: {val_generator.samples:,}\n")
    f.write(f"  - Classes: {num_classes}\n")
    f.write(f"\nModel Configuration:\n")
    f.write(f"  - Image Size: {IMG_SIZE}x{IMG_SIZE}\n")
    f.write(f"  - Batch Size: {BATCH_SIZE}\n")
    f.write(f"  - Epochs: {len(history.history['loss'])}\n")
    f.write(f"  - Learning Rate: {LEARNING_RATE}\n")
    f.write(f"\nFinal Results:\n")
    f.write(f"  - Validation Accuracy: {val_accuracy*100:.2f}%\n")
    f.write(f"  - Validation Loss: {val_loss:.4f}\n")

print("\n✓ Training summary saved to 'training_summary.txt'")
print("\nAll done! ")