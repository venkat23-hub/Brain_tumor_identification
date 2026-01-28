# Brain Tumor Identification and Classification Using Deep Learning

## 📌 Project Overview
Brain tumors are abnormal growths of cells in the brain and can be life-threatening if not detected early. Manual analysis of MRI scans is time-consuming and highly dependent on expert radiologists. This project proposes an automated system for brain tumor identification and classification using deep learning techniques.

The system analyzes MRI images and classifies them into **tumor** and **normal** categories using a Convolutional Neural Network (CNN). The proposed approach improves accuracy, reduces human effort, and supports early diagnosis.

---

## 🎯 Objectives
- To automatically detect brain tumors from MRI images  
- To classify MRI images into tumor and normal classes  
- To evaluate model performance using standard metrics  
- To reduce manual effort in medical image analysis  

---

## 🧠 Dataset
- **Source:** Kaggle  
- **Dataset Name:** Brain MRI Images for Brain Tumor Detection  
- **Classes:**
  - `yes` – Tumor images  
  - `no` – Normal images  

### Dataset Structure
- brain_tumor_dataset/
- ├── yes/
- └── no/

---

## ⚙️ Methodology
1. **Image Preprocessing**
   - Resizing images
   - Grayscale conversion
   - Normalization
   - Noise reduction

2. **Feature Learning**
   - Automatic feature extraction using CNN layers

3. **Classification**
   - CNN model with convolution, pooling, and dense layers
   - Softmax activation for final classification

4. **Evaluation**
   - Accuracy
   - Precision
   - Recall
   - F1-score

---

## 🧪 Evaluation Metrics
- **Accuracy:** Overall correctness of classification  
- **Precision:** Correctly predicted tumor cases  
- **Recall:** Ability to detect actual tumor cases  
- **F1-score:** Balance between precision and recall  

The model achieves high accuracy and reliable performance on MRI image classification.

---

## 🛠️ Technologies Used
- **Programming Language:** Python  
- **Frameworks & Libraries:**
  - TensorFlow / Keras
  - NumPy
  - OpenCV
  - Matplotlib
  - Scikit-learn
  - Pandas
- **Platform:** Google Colab / Local Machine  

---

## 📁 Project Structure
- Brain_Tumor_Identification/
- │
- ├── brain_tumor_dataset/
- │ ├── yes/
- │ └── no/
- │
- ├── brain_tumor_detection.ipynb
- ├── requirements.txt
- └── README.md

---

## 🚀 How to Run the Project
1. Clone or download the repository  
2. Upload the dataset to the project directory  
3. Install dependencies:
   ```bash
  - pip install -r requirements.txt
  - Run the Jupyter notebook or Python script

- Observe model training, evaluation metrics, and results
