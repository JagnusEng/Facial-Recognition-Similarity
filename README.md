Facial Recognition Similarity
🚀 Deep Learning-Based Face Recognition & Verification

📌 Overview
This project implements facial recognition similarity using deep learning techniques. It utilizes pre-trained VGG-Face models and DeepFace to analyze, compare, and verify faces based on their embeddings. The model processes images, extracts facial features, and determines if two faces belong to the same person.

🚀 Features
✅ Face Detection – Detects faces in images using pre-trained models.
✅ Face Embeddings – Converts faces into numerical vectors for comparison.
✅ Face Verification – Compares two faces using cosine similarity.
✅ Pre-trained Weights – Uses vgg_face_weights.h5 to improve performance.
✅ Support for Kaggle Datasets – Allows downloading datasets directly from Kaggle.

📂 Dataset
The dataset used in this project is Celebrity Face Recognition Dataset from Kaggle. You can download it using:

bash
Copy
Edit
!kaggle datasets download -d bhaveshmittal/celebrity-face-recognition-dataset -p /content/datasets/celebrity_faces --unzip
Alternatively, you can use your own images.

🛠️ Installation
1️⃣ Clone Repository
bash
Copy
Edit
git clone https://github.com/JagnusEng/JagnusEng-Facial-Recognition-Similarity.git
cd JagnusEng-Facial-Recognition-Similarity
2️⃣ Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
pip install deepface
3️⃣ Download Pre-trained Weights
bash
Copy
Edit
!gdown --id 1CPSeum3HpopfomUEK1gybeuIVoeJT_Eo -O vgg_face_weights.h5
🎯 Usage
1️⃣ Running Face Verification
python
Copy
Edit
from deepface import DeepFace
result = DeepFace.verify("img1.jpg", "img2.jpg")
print("Is Verified: ", result["verified"])
2️⃣ Face Analysis (Age, Gender, Emotion, Ethnicity)
python
Copy
Edit
DeepFace.analyze(img_path="test.jpg", actions=['age', 'gender', 'race', 'emotion'])
3️⃣ Face Recognition
python
Copy
Edit
DeepFace.find(img_path="test.jpg", db_path="./datasets/")
📸 Example Output


🛠️ Technologies Used
🖥️ Python
📸 OpenCV
🤖 TensorFlow/Keras
🏆 DeepFace
🔢 NumPy, Matplotlib
🎯 Pre-trained VGG-Face Model
📝 Credits
DeepFace Library: DeepFace GitHub
Pre-trained VGG-Face Model: VGG-Face Weights
Kaggle Dataset: Celebrity Face Recognition Dataset
🔗 Repository
GitHub Repository

🚀 Future Improvements
✅ Support for FaceNet & OpenFace
✅ Integrate with MongoDB for storing embeddings
✅ Build a Web App for real-time facial recognition

📌 Author: Jean Agnus
📅 Last Updated: March 2025
