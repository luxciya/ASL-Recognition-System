# 🤟 ASL Recognition System

An AI-based American Sign Language (ASL) recognition system that uses computer vision and deep learning to recognize hand signs from images or camera input.

## 📌 Problem

Communication can be difficult between people who use sign language and people who do not understand it.

This project explores how AI and computer vision can be used to recognize ASL hand signs and convert them into understandable labels.

## 💡 What I Built

I developed an **ASL Recognition System** using computer vision and deep learning.

The system detects hand signs and predicts the corresponding ASL class.

The project recognizes **28 different hand-sign classes**.

## 🛠️ Technologies Used

* **Python**
* **TensorFlow**
* **MediaPipe**
* **OpenCV**
* **NumPy**
* **Machine Learning**
* **Deep Learning**
* **Computer Vision**

## 🤖 How It Works

```text
Camera / Image
      ↓
Hand Detection
      ↓
MediaPipe
      ↓
Image / Landmark Processing
      ↓
Deep Learning Model
      ↓
ASL Sign Prediction
      ↓
Recognized Label
```

## 📊 Results

The model achieved approximately **98.85% validation accuracy** on the evaluated dataset.

The system was trained to recognize **28 ASL classes**.

> Model performance can vary depending on the dataset, training configuration, and input conditions.

## 🎯 My Role

I worked on:

* Preparing and processing the image data
* Hand detection using MediaPipe
* Model development and training
* Deep learning implementation using TensorFlow
* Model evaluation
* Testing ASL sign recognition
* Integrating computer vision components

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
```

### 2. Open the project

```bash
cd YOUR_REPOSITORY
```

### 3. Install the required packages

```bash
pip install -r requirements.txt
```

### 4. Run the application

Run the main Python file used by the project.

```bash
python app.py
```

> Replace `app.py` with the actual entry-point file if your project uses a different filename.

## 🔮 Future Improvements

* Recognize more ASL signs
* Improve recognition under different lighting conditions
* Add real-time sentence formation
* Add text-to-speech output
* Improve real-time camera performance
* Extend the system to support continuous sign recognition

## 👩‍💻 Project

**ASL Recognition System**

An AI and computer vision project developed using **Python, MediaPipe, TensorFlow, and OpenCV** to recognize American Sign Language hand signs.
