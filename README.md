Cancer Prediction using Logistic Regression
Overview
This project aims to predict whether a tumor is malignant (M) or benign (B) based on various features like radius, texture, perimeter, area, smoothness, and others. The dataset used for this project is the Breast Cancer Wisconsin (Diagnostic) dataset, which contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.

The model used for this prediction is Logistic Regression, which is trained on a subset of the dataset and evaluated using accuracy and other classification metrics.

Dataset
The dataset used in this project is from the UCI Machine Learning Repository and is available here.

Columns in the dataset:
id: Unique identifier for each sample.
diagnosis: The diagnosis of the tumor (M = malignant, B = benign).
radius_mean, texture_mean, perimeter_mean, area_mean, etc.: Various features computed from the image of the tumor.
Libraries Used
Pandas: Data manipulation and analysis.
NumPy: Mathematical functions.
Scikit-learn: Machine learning algorithms and utilities for model training and evaluation.
Steps to Run the Project
Clone this repository to your local machine:

bash
Copy code
git clone https://github.com/yourusername/cancer-prediction.git
Install the required libraries:

bash
Copy code
pip install -r requirements.txt
Run the Python script:

bash
Copy code
python cancer_prediction.py
The model will be trained and evaluated, and the accuracy and classification report will be printed.

Model Performance
After training the model, the following results were obtained on the test set:

Accuracy: 96.49%
Confusion Matrix:
lua
Copy code
[[64,  2],
 [ 2, 46]]
Classification Report:
css
Copy code
            precision    recall  f1-score   support
         B       0.97      0.97      0.97        66
         M       0.96      0.96      0.96        48
  accuracy                           0.96       114
 macro avg       0.96      0.96      0.96       114
weighted avg 0.96 0.96 0.96 114
