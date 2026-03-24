import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Parameters
BATCH_SIZE = 64
EPOCHS = 30
# Lower learning rate for fine-tuning to avoid catastrophic forgetting
LEARNING_RATE = 1e-5
NUM_CLASSES = 6
DATA_DIR = 'data/organized'

# CRITICAL: We must maintain the exact same class indices as the original model
# The original model mapped: 0: Angry, 1: Fear, 2: Happy, 3: Sad, 4: Surprise, 5: Neutral
CLASS_NAMES = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

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
    plt.title('Training and Validation Accuracy (Fine-Tuning)')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss (Fine-Tuning)')
    plt.savefig('reports/finetune_history.png')
    plt.show()

def finetune():
    if not os.path.exists(DATA_DIR):
        print(f"Error: Organized dataset not found at {DATA_DIR}")
        return

    # Data Augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=0.2 # 80/20 split
    )

    # Validation generator (only rescaling)
    valid_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    # Load data from directory
    print("Loading training data...")
    train_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=(48, 48), # Original model input shape (48, 48, 1)
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASS_NAMES, # Crucial step to map folders to exactly the same IDs
        subset='training',
        shuffle=True
    )

    print("Loading validation data...")
    validation_generator = valid_datagen.flow_from_directory(
        DATA_DIR,
        target_size=(48, 48),
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASS_NAMES,
        subset='validation',
        shuffle=False
    )
    
    # Load the best pre-trained model
    model_path = 'Models/emotion_model_final_v1.keras'
    if not os.path.exists(model_path):
        # Fallback to older model if final_v1 doesn't exist
        model_path = 'Models/emotion_model_nb.keras'
        if not os.path.exists(model_path):
            print("Error: Pre-trained model not found. Cannot fine-tune.")
            return
            
    print(f"Loading base model from {model_path}...")
    model = load_model(model_path)
    
    # Recompile with a very low learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Setup Callbacks
    # Save the best model during fine-tuning
    checkpoint = ModelCheckpoint(
        'Models/emotion_model_finetuned.keras', 
        monitor='val_accuracy', 
        verbose=1, 
        save_best_only=True, 
        mode='max'
    )
    
    # Reduce learning rate further if validation loss plateaus
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=3, 
        min_lr=1e-7, 
        verbose=1,
        mode='min'
    )
    
    # Stop early if no improvement
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=8, 
        verbose=1,
        mode='min', 
        restore_best_weights=True
    )

    callbacks_list = [checkpoint, reduce_lr, early_stop]

    # Train model
    print("Starting fine-tuning...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=callbacks_list
    )
    
    # Save final trained model
    model.save('Models/emotion_model_finetuned_final.keras')
    print("Fine-tuning completed. Model saved as 'Models/emotion_model_finetuned_final.keras'")

    # Make sure reports directory exists
    os.makedirs('reports', exist_ok=True)
    
    # Plot results
    plot_training_history(history)

if __name__ == "__main__":
    finetune()
