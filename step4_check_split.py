from tensorflow.keras.preprocessing.image import ImageDataGenerator

BASE_DIR = "D:\projectpcd\python10\percobaan_lagi2\BISINDO_split"
img_size = 64
batch_size = 8

datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(
    BASE_DIR + "/train",
    target_size=(img_size, img_size),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="categorical"
)

val_data = datagen.flow_from_directory(
    BASE_DIR + "/val",
    target_size=(img_size, img_size),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="categorical"
)

test_data = datagen.flow_from_directory(
    BASE_DIR + "/test",
    target_size=(img_size, img_size),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False
)

print("Train:", train_data.samples)
print("Val  :", val_data.samples)
print("Test :", test_data.samples)
