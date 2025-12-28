import os
import shutil
import random

# PATH (sesuaikan)
SOURCE_DIR = "D:\projectpcd\python10\percobaan_lagi2\dataset_raw"
TARGET_DIR = "D:\projectpcd\python10\percobaan_lagi2\BISINDO_split"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

os.makedirs(TARGET_DIR, exist_ok=True)

for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(TARGET_DIR, split), exist_ok=True)

for label in os.listdir(SOURCE_DIR):
    label_path = os.path.join(SOURCE_DIR, label)
    if not os.path.isdir(label_path):
        continue

    images = os.listdir(label_path)
    random.shuffle(images)

    total = len(images)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    splits = {
        'train': images[:train_end],
        'val': images[train_end:val_end],
        'test': images[val_end:]
    }

    for split, files in splits.items():
        split_label_dir = os.path.join(TARGET_DIR, split, label)
        os.makedirs(split_label_dir, exist_ok=True)

        for file in files:
            src = os.path.join(label_path, file)
            dst = os.path.join(split_label_dir, file)
            shutil.copy(src, dst)

print("Split dataset selesai")
