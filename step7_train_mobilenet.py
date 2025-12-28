from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from step5_train_val_datagen import train_data, val_data
from step6_mobilenet_model import model
from step7_plot_training import plot_training

callbacks = [
    EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
    ModelCheckpoint("model_bisindo_mobilenet.h5", save_best_only=True)
]

history = model.fit(
    train_data,
    epochs=30,
    validation_data=val_data,
    callbacks=callbacks
)

plot_training(history)
