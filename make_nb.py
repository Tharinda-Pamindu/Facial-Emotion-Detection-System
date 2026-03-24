import nbformat as nbf

nb = nbf.v4.new_notebook()

# Cell 1: Markdowh Header
text = """# Emotion Detection Model Fine-Tuning
This notebook fine-tunes the base emotion model using the new organized dataset with data augmentation."""
nb.cells.append(nbf.v4.new_markdown_cell(text))

# Cell 2: Imports and Constants
code1 = """import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Parameters
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-5
NUM_CLASSES = 6
DATA_DIR = 'data/organized'

# Consistent Class Mapping
CLASS_NAMES = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']"""
nb.cells.append(nbf.v4.new_code_cell(code1))

# Cell 3: Data Generators setup
code2 = """train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

valid_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

print("Loading training data...")
train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(48, 48),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASS_NAMES,
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
)"""
nb.cells.append(nbf.v4.new_markdown_cell("## Setup Data Generators"))
nb.cells.append(nbf.v4.new_code_cell(code2))

# Cell 4: Model loading
code3 = """model_path = 'Models/emotion_model_final_v1.keras'
if not os.path.exists(model_path):
    model_path = 'Models/emotion_model_nb.keras'

print(f"Loading base model from {model_path}...")
model = load_model(model_path)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()"""
nb.cells.append(nbf.v4.new_markdown_cell("## Load and Compile Base Model"))
nb.cells.append(nbf.v4.new_code_cell(code3))

# Cell 5: Callbacks and Fit
code4 = """checkpoint = ModelCheckpoint(
    'Models/emotion_model_finetuned.keras', 
    monitor='val_accuracy', 
    verbose=1, 
    save_best_only=True, 
    mode='max'
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, 
    patience=3, 
    min_lr=1e-7, 
    verbose=1,
    mode='min'
)

early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=8, 
    verbose=1,
    mode='min', 
    restore_best_weights=True
)

callbacks_list = [checkpoint, reduce_lr, early_stop]

print("Starting fine-tuning...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=callbacks_list
)

model.save('Models/emotion_model_finetuned_final.keras')
print("Fine-tuning completed. Model saved.")"""
nb.cells.append(nbf.v4.new_markdown_cell("## Train Model"))
nb.cells.append(nbf.v4.new_code_cell(code4))

# Cell 6: Plotting
code5 = """acc = history.history['accuracy']
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

os.makedirs('reports', exist_ok=True)
plt.savefig('reports/finetune_history.png')
plt.show()"""
nb.cells.append(nbf.v4.new_markdown_cell("## Plot and Save History"))
nb.cells.append(nbf.v4.new_code_cell(code5))

with open('Finetune_Emotion_Model.ipynb', 'w') as f:
    nbf.write(nb, f)
