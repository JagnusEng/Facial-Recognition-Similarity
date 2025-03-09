## Facial Recognition Similarity

Deep Learning-Based Face Recognition & Verification

Overview
This project implements facial recognition similarity using deep learning techniques. It utilizes pre-trained VGG-Face models and DeepFace to analyze, compare, and verify faces based on their embeddings. The model processes images, extracts facial features, and determines if two faces belong to the same person.

## Features

- Face Detection – Detects faces in images using pre-trained models.
- Face Embeddings – Converts faces into numerical vectors for comparison.
- Face Verification – Compares two faces using cosine similarity.
- Pre-trained Weights – Uses vgg_face_weights.h5 to improve performance.
- Support for Kaggle Datasets – Allows downloading datasets directly from Kaggle.

## Dataset

!kaggle datasets download -d bhaveshmittal/celebrity-face-recognition-dataset -p /content/datasets/celebrity_faces --unzip


## Clone Repository

git clone https://github.com/JagnusEng/JagnusEng-Facial-Recognition-Similarity.git
cd JagnusEng-Facial-Recognition-Similarity

## Install Dependencies

pip install -r requirements.txt
pip install deepface


## Download Pre-trained Weights

!gdown --id 1CPSeum3HpopfomUEK1gybeuIVoeJT_Eo -O vgg_face_weights.h5


## Usage

from deepface import DeepFace
result = DeepFace.verify("img1.jpg", "img2.jpg")
print("Is Verified: ", result["verified"])


## Face Analysis (Age, Gender, Emotion, Ethnicity)

DeepFace.analyze(img_path="test.jpg", actions=['age', 'gender', 'race', 'emotion'])


## Face Recognition

DeepFace.find(img_path="test.jpg", db_path="./datasets/")


## Technologies Used

- 🖥️ Python
- 📸 OpenCV
- 🤖 TensorFlow/Keras
- 🏆 DeepFace
- 🔢 NumPy, Matplotlib
- 🎯 Pre-trained VGG-Face Model

## Credits

DeepFace Library: DeepFace GitHub
Pre-trained VGG-Face Model: VGG-Face Weights
Kaggle Dataset: Celebrity Face Recognition Dataset

## License

- This project is licensed under the MIT License. You are free to use, copy, modify, and distribute this software as long as the original copyright notice and this license are included in any distribution.
