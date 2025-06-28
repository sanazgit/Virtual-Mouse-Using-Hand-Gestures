# Virtual Mouse Using Hand Gestures

This project implements a **virtual mouse system** controlled through hand gestures detected by a webcam. It combines computer vision (via MediaPipe), deep learning (a CNN model), and automation tools to simulate mouse actions such as movement and clicks — all without touching a physical mouse.

---

##  Project Structure

| File | Description |
|------|-------------|
| `VirtualMous.ipynb` | Main notebook for running the virtual mouse system |
| `posedetect.ipynb` | Notebook for training the CNN model on the hand gesture dataset |
| `model.py` | Python script containing the CNN model architecture |
| `slm_4class.pth` | Trained model weights (used in `VirtualMous.ipynb`) |

---

##  Model Description

The model used is a custom **Convolutional Neural Network (CNN)** designed to classify grayscale hand gesture images into four classes:

- Move Left  
- Move Down  
- Right Click  
- Double Click  

Architecture Highlights:
- Multiple convolutional layers with BatchNorm, ReLU, Dropout
- MaxPooling and AdaptiveAvgPooling
- Final linear classifier

See `model.py` for full implementation.

---

##  Dataset

The dataset used is [**SL-MNIST**](https://www.kaggle.com/datasets/datamunge/sign-language-mnist), a modified version of the original MNIST-style dataset tailored for **Sign Language / Hand Gesture classification**.

Each image:
- Represents a single hand gesture
- Is preprocessed to 28x28 grayscale
- Categorized into 4 action classes suitable for mouse control

You can download or generate this dataset by referring to the links mentioned inside `posedetect.ipynb`.

---

## ▶️ How to Run

1. Make sure you have the following dependencies installed:
   ```bash
   pip install mediapipe opencv-python torch torchvision pyautogui
