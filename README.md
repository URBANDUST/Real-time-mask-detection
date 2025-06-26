# 🧠 Real-Time Face Mask Detection System

A real-time **Face Mask Detection System** built using **Python**, **OpenCV**, and a **pre-trained TensorFlow/Keras model**. It uses your webcam to detect faces and classify whether each person is wearing a mask or not.

---

## 🔧 Features

- Detects human faces using Haar Cascade classifiers.
- Classifies each face as **"Mask"** or **"No Mask"** using a deep learning model.
- Shows bounding boxes and confidence scores in real-time.
- Saves output video with detection results.
- Configurable confidence threshold and model settings.

---

## 🖥️ Demo

https://user-images.githubusercontent.com/your-username/demo-mask-detection.gif

---

## 📁 Project Structure

mask_detection_project/

├── haarcascade_frontalface_default.xml

├── mask_detector.model # Pre-trained Keras model (.h5)

├── mask_detection.py # Main Python script

├── output_mask_detection.avi # Output video (generated on run)

└── README.md


---

## 🔍 Requirements

- Python 3.8 / 3.9 / 3.10  
- Compatible with TensorFlow 2.x

### 📦 Install Dependencies

Use a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install --upgrade pip
pip install tensorflow opencv-python numpy
