# Breast Cancer Detection using BreaKHis Dataset

## Overview
This project focuses on building a deep learning model to detect breast cancer using the BreaKHis dataset. The dataset consists of histopathological images of breast tissue samples, categorized into benign and malignant classes.

## Dataset
- **Source**: [BreaKHis Dataset on Kaggle](https://www.kaggle.com/datasets/ambarish/breakhis)
- **Description**: The dataset contains images of breast tissue samples categorized into benign and malignant classes. Each image is annotated with its corresponding class label.

## Project Structure
- **dataset**: Directory containing the BreaKHis dataset split into train and validation sets.
- **Breast_Cancer_Detection.ipynb**: Jupyter notebook containing the Python code for training the deep learning model using TensorFlow and Keras. This can also be access by [Google Colab](Google Colab).

- **app.py**: The flask server to get the UI.
- **requirements.txt**: List of Python dependencies required to run the project.
- **README.md**: This file, providing an overview of the project, dataset, and instructions.

## Model Architecture
- **Base Model**: Transfer learning using ResNet50 pre-trained on ImageNet.
- **Top Layers**: Global Average Pooling, Dense layers with ReLU activation, Dropout, and Sigmoid output for binary classification.

## Training
- **Data Augmentation**: Applied to the training set for improved generalization.
- **Training Procedure**: Model trained using Adam optimizer, binary cross-entropy loss, and early stopping based on validation loss.

## Evaluation
- **Metrics**: Evaluated on the validation set using accuracy and loss metrics.
- **Confusion Matrix**: Used to analyze model performance on class predictions.

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/breast-cancer-detection.git
   cd breast-cancer-detection
2. Start the flask server:
   ```bash
   pip install -r requirements.txt
   python app.py
3. Enjoy:
   ```bash
   visit http://127.0.0.1:5000/

