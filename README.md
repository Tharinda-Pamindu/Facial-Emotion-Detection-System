<div><img src="/assets/ss-1.png" alt="FR" /></div>
# 🎭 Facial Emotion Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

**A real-time facial emotion detection system using Convolutional Neural Networks (CNN) trained on the FER-2013 dataset.**

_Classifies 6 emotions: 😠 Angry · 😨 Fear · 😄 Happy · 😢 Sad · 😲 Surprise · 😐 Neutral_

---

</div>

## 📋 Table of Contents

- [Overview](#-overview)
- [Emotion Classes](#-emotion-classes)
- [Project Structure](#-project-structure)
- [Preprocessing Pipeline](#-preprocessing-pipeline)
- [Model Architecture](#-model-architecture)
- [Setup & Installation](#-setup--installation)
- [How to Run](#-how-to-run)
- [Team](#-team)

---

## 🔍 Overview

This project uses deep learning to detect and classify facial emotions in real-time through a webcam feed. It was developed for the **ICT3212 - Introduction to Intelligent Systems** module at Rajarata University of Sri Lanka.

| Feature       | Details                    |
| ------------- | -------------------------- |
| **Dataset**   | FER-2013 (Kaggle)          |
| **Model**     | Custom CNN (4 Conv blocks) |
| **Input**     | 48×48 grayscale images     |
| **Classes**   | 6 emotions                 |
| **Interface** | Streamlit web app          |
| **Detection** | OpenCV Haar Cascade        |

---

## 🎯 Emotion Classes

| Label | Emotion     |     Samples | Percentage |
| :---: | ----------- | ----------: | ---------: |
|   0   | 😠 Angry    |      ~4,953 |      14.0% |
|   1   | 😨 Fear     |      ~5,121 |      14.5% |
|   2   | 😄 Happy    |      ~8,989 |      25.4% |
|   3   | 😢 Sad      |      ~6,077 |      17.2% |
|   4   | 😲 Surprise |      ~4,002 |      11.3% |
|   5   | 😐 Neutral  |      ~6,198 |      17.5% |
|       | **Total**   | **~35,340** |   **100%** |

> **Note:** The original Disgust class was removed due to insufficient samples (~547).

---

## 📁 Project Structure

```
Facial-Emotion-Detection-System/
│
├── 📓 Notebooks
│   ├── Facial_Emotion_Detection.ipynb    # Full pipeline (train + inference)
│   ├── Dataset_Exploration.ipynb         # Dataset organization & analysis
│   └── Image_Preprocessing.ipynb         # Preprocessing steps
│
├── 📂 src/
│   ├── data_preprocessing.py             # Data loading & processing
│   ├── model.py                          # CNN architecture definition
│   ├── train.py                          # Training script
│   └── app.py                            # Streamlit real-time app
│
├── 📂 data/
│   ├── fer2013.csv                       # FER-2013 dataset (download separately)
│   └── organized/                        # Class-wise image folders (auto-generated)
│       ├── Angry/
│       ├── Fear/
│       ├── Happy/
│       ├── Sad/
│       ├── Surprise/
│       └── Neutral/
│
├── 📂 Models/                            # Saved trained models
├── 📄 requirements.txt                   # Dependencies
├── 📄 Model_Card.md                      # Hugging Face model card
├── 📄 Dataset_Exploration_Report.txt     # Dataset analysis report
├── 📄 Image_Preprocessing_Report.txt     # Preprocessing report
├── 📄 Project_Report.txt                 # Full project report
└── 📄 README.md
```

---

## ⚙️ Preprocessing Pipeline

```
 ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
 │  1. RESIZE  │────▶│ 2. NORMALIZE│────▶│ 3. TENSOR   │────▶│  4. SPLIT   │
 │  48 × 48    │     │  [0, 1]     │     │ (N,48,48,1) │     │ 80/20       │
 └─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

| Step | Operation         | Before            | After                     |
| ---- | ----------------- | ----------------- | ------------------------- |
| 1    | Resize            | Raw CSV pixels    | 48 × 48 images            |
| 2    | Normalize         | [0, 255] uint8    | [0.0, 1.0] float32        |
| 3    | Convert to Tensor | NumPy (N, 48, 48) | TF Tensor (N, 48, 48, 1)  |
| 4    | Train/Test Split  | 35,340 total      | 28,272 Train / 7,068 Test |

---

## 🧠 Model Architecture

```
Input (48 × 48 × 1)
       │
       ▼
┌──────────────────────┐
│ Conv2D(64) + BN + MP │──── Block 1
│ + Dropout(0.25)      │
└──────────┬───────────┘
           │
┌──────────▼───────────┐
│ Conv2D(128) + BN + MP│──── Block 2
│ + Dropout(0.25)      │
└──────────┬───────────┘
           │
┌──────────▼───────────┐
│ Conv2D(512) + BN + MP│──── Block 3
│ + Dropout(0.25)      │
└──────────┬───────────┘
           │
┌──────────▼───────────┐
│ Conv2D(512) + BN + MP│──── Block 4
│ + Dropout(0.25)      │
└──────────┬───────────┘
           │
    ┌──────▼──────┐
    │   Flatten   │
    └──────┬──────┘
           │
  ┌────────▼────────┐
  │ Dense(256) + BN │
  │ + Dropout(0.5)  │
  └────────┬────────┘
           │
  ┌────────▼────────┐
  │ Dense(512) + BN │
  │ + Dropout(0.5)  │
  └────────┬────────┘
           │
    ┌──────▼──────┐
    │ Dense(6)    │
    │ Softmax     │
    └─────────────┘
```

**Training Configuration:**

- **Optimizer:** Adam (lr = 0.0001)
- **Loss:** Categorical Crossentropy
- **Callbacks:** ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

---

## 🚀 Setup & Installation

### Prerequisites

- Python 3.8+
- Webcam (for real-time detection)
- [FER-2013 Dataset](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Tharinda-Pamindu/Facial-Emotion-Detection-System.git
cd Facial-Emotion-Detection-System

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Download FER-2013 dataset and place it in:
#    data/fer2013.csv
```

---

## ▶️ How to Run

### Option 1: Jupyter Notebooks (Recommended)

```bash
jupyter notebook
```

Run the notebooks in order:

1. **`Dataset_Exploration.ipynb`** — Explore & organize the dataset
2. **`Image_Preprocessing.ipynb`** — Preprocess images
3. **`Facial_Emotion_Detection.ipynb`** — Train model & run detection

### Option 2: Python Scripts

```bash
# Train the model (20-60 min depending on hardware)
python src/train.py

# Run the Streamlit app
streamlit run src/app.py
```

### Option 3: Batch Scripts (Windows)

```bash
# Train
train_model.bat

# Run app
run_app.bat
```

---

## 👥 Team — Inflators

| Member              | Role            |
| ------------------- | --------------- |
| DTPD Wickramasinghe | 👑 Group Leader |
| DVTR Vitharana      | Member          |
| RSR Ranathunga      | Member          |
| DDSS Kumasaru       | Member          |
| SHD Mihidumpita     | Member          |

---

<div align="center">

**ICT3212 - Introduction to Intelligent Systems**
Rajarata University of Sri Lanka

</div>
