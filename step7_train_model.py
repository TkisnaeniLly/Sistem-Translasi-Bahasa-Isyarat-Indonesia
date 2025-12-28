from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from step5_train_val_datagen import train_data, val_data
from step6_build_model import model

epochs = 50

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=8,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        'model_bisindo_mobilenet.h5',
        monitor='val_loss',
        save_best_only=True
    )
]

history = model.fit(
    train_data,
    epochs=epochs,
    validation_data=val_data,
    callbacks=callbacks
)
from step7_plot_training import plot_training
plot_training(history)
