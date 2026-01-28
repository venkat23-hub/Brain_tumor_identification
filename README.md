# Brain Tumor Identification and Classification Using Deep Learning

## Project Overview
Brain tumors are abnormal growths of cells in the brain and can be life-threatening if not detected early. Manual analysis of MRI scans is time-consuming and highly dependent on expert radiologists. This project presents an automated system for brain tumor identification and classification using deep learning techniques.

The system analyzes MRI images and classifies them into tumor and normal categories using a Convolutional Neural Network (CNN). The proposed approach improves accuracy, reduces human effort, and supports early diagnosis.

## Objectives
- Automatically detect brain tumors from MRI images
- Classify MRI images into tumor and normal classes
- Evaluate model performance using standard metrics
- Reduce manual effort in medical image analysis

## Dataset
Source: Kaggle  
Dataset Name: Brain MRI Images for Brain Tumor Detection

Classes:
- yes: Tumor images
- no: Normal images

Dataset structure:
- brain_tumor_dataset/
- ├── yes/
- └── no/

## Methodology
1. Image preprocessing including resizing, grayscale conversion, normalization, and noise reduction.
2. Automatic feature extraction using convolutional layers.
3. Classification using a CNN with convolution, pooling, and fully connected layers.
4. Performance evaluation using standard classification metrics.

## Evaluation Metrics
- Accuracy: Measures the overall correctness of the model.
- Precision: Measures how many predicted tumor cases are correct.
- Recall: Measures the ability to detect actual tumor cases.
- F1-score: Balances precision and recall.

## Technologies Used
- Python
- TensorFlow and Keras
- NumPy
- OpenCV
- Matplotlib
- Scikit-learn
- Pandas

## Project Structure
Brain_Tumor_Identification/
- ├── brain_tumor_dataset/
- │ ├── yes/
- │ └── no/
- ├── brain_tumor_detection.ipynb
- ├── requirements.txt
- └── README.md

## How to Run the Project
1. Clone or download the project repository.
2. Upload the dataset to the project directory.
3. Install dependencies using the command:
   pip install -r requirements.txt
4. Run the Jupyter notebook or Python script.
5. Observe the training process and evaluation results.

## Results
The proposed system achieves high accuracy in detecting brain tumors from MRI images and demonstrates reliable classification performance.

## Deployment
The trained brain tumor classification model has been deployed as a web application using Streamlit. The web app allows users to upload MRI images, automatically preprocess them, and obtain real-time predictions indicating whether a brain tumor is present or not.

Deployment Link:  
https://braintumoridentification.streamlit.app/

## Future Scope
- Extension to multi-class tumor classification.
- Use of 3D MRI images for improved analysis.
- Integration into real-time clinical systems.

## Academic Note
This project is developed for academic and research purposes as part of an undergraduate engineering program.
