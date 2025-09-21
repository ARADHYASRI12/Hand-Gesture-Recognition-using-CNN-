# Hand-Gesture-Recognition-using-CNN-
Hand Gesture Recognition & Energy Forecasting

This repository contains two machine learning projects implemented as part of coursework and self-driven exploration:

Hand Gesture Recognition using Convolutional Neural Networks (CNN)

A deep learning model trained on the Sign Language MNIST dataset
.

Classifies American Sign Language (A-Z, excluding J & Z).

├── hand_gesture_cnn.py         # Hand Gesture Recognition (CNN)
├── energy_forecasting_lstm.py  # Energy Consumption Forecasting (LSTM)
├── HandSignRecog.h5            # Saved CNN model
├── model_architecture.png      # CNN architecture diagram
├── README.md                   # Documentation
└── requirements.txt            # Python dependencies

⚙️ Installation & Setup

Clone the repository:

git clone https://github.com/your-username/gesture-energy-ml.git
cd gesture-energy-ml


Create a virtual environment:

python -m venv venv
source venv/bin/activate   # for Linux/Mac
venv\Scripts\activate      # for Windows


Install dependencies:

pip install -r requirements.txt

🖐️ Hand Gesture Recognition (CNN)
Training
python hand_gesture_cnn.py

Dataset

Uses Sign Language MNIST dataset (sign_mnist_train.csv).

Normalized & reshaped to 28×28 grayscale images.

Features

CNN with 2 Conv + MaxPool layers

Dense layer (128 units, ReLU) + Dropout

Output layer: 26 classes (softmax)

Accuracy: 99%+ on validation set

Custom Prediction

Replace 1127_C.jpg with your own hand gesture image:

image_path = 'your_image.jpg'
predict_image(model, image_path, class_labels)
