from tensorflow.keras.preprocessing.image import ImageDataGenerator

BASE_DIR = "D:/projectpcd/python10/percobaan_lagi2/BISINDO_split"
img_size = 224
batch_size = 8

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=8,
    zoom_range=0.1,
    width_shift_range=0.05,
    height_shift_range=0.05,
    brightness_range=[0.9, 1.1],
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    BASE_DIR + "/train",
    target_size=(img_size, img_size),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical"
)

val_data = val_test_datagen.flow_from_directory(
    BASE_DIR + "/val",
    target_size=(img_size, img_size),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical"
)

test_data = val_test_datagen.flow_from_directory(
    BASE_DIR + "/test",
    target_size=(img_size, img_size),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False
)
