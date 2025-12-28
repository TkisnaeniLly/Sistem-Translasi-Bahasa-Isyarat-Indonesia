import os

dataset_path = "D:\projectpcd\python10\percobaan_lagi2\dataset_raw"  # SESUAIKAN PATH KAMU

classes = os.listdir(dataset_path)

print("Jumlah kelas:", len(classes))
print("Daftar kelas:", classes)

for cls in classes:
    cls_path = os.path.join(dataset_path, cls)
    if os.path.isdir(cls_path):
        print(f"Huruf {cls}: {len(os.listdir(cls_path))} gambar")
