import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tensorflow.keras.models import load_model
from step5_train_val_datagen import test_data

# =============================
# LOAD MODEL
# =============================
model = load_model("model_bisindo_mobilenet.h5")

# =============================
# PREDIKSI
# =============================
test_data.reset()
y_true = test_data.classes
y_pred_prob = model.predict(test_data)
y_pred = np.argmax(y_pred_prob, axis=1)

class_names = list(test_data.class_indices.keys())

# =============================
# CONFUSION MATRIX
# =============================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(14, 12))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",          # fokus & nyaman dibaca
    xticklabels=class_names,
    yticklabels=class_names,
    annot_kws={"size": 10, "color": "black"}
)

plt.title("Confusion Matrix BISINDO (MobileNetV2)", fontsize=14)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()

# =============================
# SAVE PNG
# =============================
plt.savefig("confusion_matrix_step8.png", dpi=300)
plt.show()
