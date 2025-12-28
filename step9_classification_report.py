from sklearn.metrics import classification_report
import numpy as np
from tensorflow.keras.models import load_model
from step5_train_val_datagen import test_data

model = load_model("model_bisindo_mobilenet.h5")

y_true = test_data.classes
y_pred = np.argmax(model.predict(test_data), axis=1)

labels = list(test_data.class_indices.keys())

report = classification_report(
    y_true, y_pred, target_names=labels
)

print(report)

with open("classification_report.txt", "w") as f:
    f.write(report)
