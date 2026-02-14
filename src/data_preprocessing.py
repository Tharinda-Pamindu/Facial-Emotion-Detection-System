import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import os

def load_fer2013(file_path):
    """
    Loads FER-2013 dataset from CSV file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}. Please download 'fer2013.csv' from Kaggle and place it in the 'data' directory.")

    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    """
    Preprocesses the data:
    - Filters out 'Disgust' (1) as per project requirements (6 classes).
    - Converts pixels string to numpy array.
    - Reshapes to (48, 48, 1).
    - Normalizes pixel values.
    - One-hot encodes labels.
    """
    # Filter out Disgust (1)
    # 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
    # We want: Happiness, Sadness, Anger, Surprise, Fear, Neutral
    # Mapping might be different in proposal but let's stick to valid subsets.
    # Proposal: Happiness, Sadness, Anger, Surprise, Fear, Neutral
    # FER2013: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
    
    # Remove Disgust
    data = data[data['emotion'] != 1]
    
    # Remap labels to 0-5
    # Current: 0, 2, 3, 4, 5, 6
    # New: 0->0 (Anger), 2->1 (Fear), 3->2 (Happy), 4->3 (Sad), 5->4 (Surprise), 6->5 (Neutral)
    label_map = {0: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
    data['emotion'] = data['emotion'].map(label_map)
    
    pixels = data['pixels'].tolist()
    emotions = pd.get_dummies(data['emotion']).values
    
    faces = []
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.asarray(face).reshape(48, 48)
        faces.append(face.astype('float32'))
    
    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1) # (N, 48, 48, 1)
    
    # Normalize
    faces /= 255.0
    
    return faces, emotions

def get_data(file_path='data/fer2013.csv'):
    """
    Main function to get train/test data.
    """
    print("Loading data...")
    data = load_fer2013(file_path)
    
    print("Preprocessing data...")
    faces, emotions = preprocess_data(data)
    
    # Split into train and test
    # FER-2013 has 'Usage' column: Training, PublicTest, PrivateTest
    # We can use that or just split randomly. The proposal says 80/20 split.
    # Let's respect the csv structure if possible, but manual split is also fine.
    
    # Using 'Usage' column
    train_data = data[data['Usage'] == 'Training']
    test_data = data[data['Usage'] != 'Training'] # PublicTest + PrivateTest
    
    # Recalculate processed arrays for split
    # Actually it's more efficient to split the processed arrays.
    # But filtering changed indices.
    
    # Let's just use train_test_split on the whole processed set to match 80/20 exactly as requested?
    # Proposal: "Splitting data into 80% Training and 20% Testing sets."
    
    X_train, X_test, y_train, y_test = train_test_split(faces, emotions, test_size=0.2, random_state=42)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    try:
        get_data()
        print("Data preprocessing successful!")
    except Exception as e:
        print(f"Error: {e}")
