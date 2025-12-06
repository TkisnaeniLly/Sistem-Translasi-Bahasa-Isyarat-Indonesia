# train_multimodal.py
import os, glob, numpy as np, cv2, joblib, tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, Model

# ------------------ PARAMETER ------------------
IMG_SIZE   = 128
SENSOR_LEN = 8                # sesuaikan dengan rf_model
BATCH      = 32
EPOCHS     = 25
CLASS      = ['A','B','C','D','E','F','G','H','I','J']  # ganti dengan yg ada
SENSOR_DIR = 'dataset/sensor'
IMAGE_DIR  = 'dataset/image'
# ----------------------------------------------

le = LabelEncoder()
le.fit(CLASS)

# 1. generator pasangan (sensor, image, label)
def parse_pair(sensor_path, img_path, label):
    # sensor
    sensor = np.loadtxt(sensor_path, delimiter=',')   # (SENSOR_LEN,)
    sensor = sensor.astype(np.float32)
    # image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.
    # label
    label_id = le.transform([label])[0]
    return sensor, img, tf.one_hot(label_id, len(CLASS))

def list_pairs(split='train'):
    sensors, images, labels = [], [], []
    for cls in CLASS:
        s_files = sorted(glob.glob(os.path.join(SENSOR_DIR, cls, '*')))
        i_files = sorted(glob.glob(os.path.join(IMAGE_DIR,  cls, '*')))
        for s, i in zip(s_files, i_files):
            sensors.append(s)
            images.append(i)
            labels.append(cls)
    return sensors, images, labels

s_list, i_list, l_list = list_pairs()
ds = tf.data.Dataset.from_tensor_slices((s_list, i_list, l_list))
ds = ds.shuffle(2048).map(lambda s,i,l: tf.py_function(
        parse_pair, [s,i,l], [tf.float32, tf.float32, tf.float32]),
        num_parallel_calls=tf.data.AUTOTUNE)
ds = ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)

# 2. arsitektur multi-modal
sensor_in = layers.Input(shape=(SENSOR_LEN,), name='sensor_in')
x1 = layers.Dense(64, activation='relu')(sensor_in)
x1 = layers.Dense(32, activation='relu')(x1)

img_in = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='img_in')
x2 = layers.Conv2D(32,3,activation='relu')(img_in)
x2 = layers.MaxPool2D()(x2)
x2 = layers.Conv2D(64,3,activation='relu')(x2)
x2 = layers.MaxPool2D()(x2)
x2 = layers.Flatten()(x2)
x2 = layers.Dense(64, activation='relu')(x2)

concat = layers.Concatenate()([x1, x2])
out = layers.Dense(len(CLASS), activation='softmax')(concat)

model = Model([sensor_in, img_in], out)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# 3. train
model.fit(ds, epochs=EPOCHS)

# 4. save
model.save('multimodal.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open('multimodal.tflite','wb').write(tflite_model)
print('Saved multimodal.h5 & multimodal.tflite')