# Facial Emotion Detection System

## Project Overview

This project uses a specific Convolutional Neural Network (CNN) to detect facial emotions in real-time.

## Prerequisites

1.  **Python 3.8+**
2.  **Virtual Environment** (Recommended)
3.  **FER-2013 Dataset** (Required for training)
    - Download `fer2013.csv` from Kaggle.
    - Place it in: `data/fer2013.csv`.

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

### 1. Train the Model

You must train the model at least once before running the application.

```bash
python src/train.py
```

- This will load the data, train the CNN, and save the model to `Models/emotion_model_final.keras`.
- **Note:** Training can take some time (20-60 minutes depending on hardware).

### 2. Run the Application

Once the model is trained, start the real-time detection app:

```bash
streamlit run src/app.py
```

- A web browser will open.
- Click **"Use Webcam"** in the sidebar to start detection.

## Project Structure

- `src/data_preprocessing.py`: Handles data loading and processing.
- `src/model.py`: Defines the CNN architecture.
- `src/train.py`: Training script.
- `src/app.py`: Real-time inference application.
- `Models/`: Directory where trained models are saved.
- `data/`: Directory for the dataset.
