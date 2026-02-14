import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from data_preprocessing import get_data
from model import build_model
import matplotlib.pyplot as plt
import os

# Parameters
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.0001
NUM_CLASSES = 6
DATA_PATH = 'data/fer2013.csv'

def train():
    # Load and preprocess data
    try:
        X_train, y_train, X_test, y_test = get_data(DATA_PATH)
    except FileNotFoundError as e:
        print(e)
        return

    # Build model
    model = build_model(input_shape=(48, 48, 1), num_classes=NUM_CLASSES)
    model.summary()

    # Callbacks
    checkpoint = ModelCheckpoint('Models/emotion_model.keras', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, mode='min')
    early_stop = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)

    callbacks_list = [checkpoint, reduce_lr, early_stop]

    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks_list,
        shuffle=True
    )
    
    # Save final model
    model.save('Models/emotion_model_final.keras')

    # Plot results
    plot_training_history(history)

def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

if __name__ == "__main__":
    train()
