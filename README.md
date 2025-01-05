# Cancer Prediction Using Logistic Regression

This project uses machine learning to predict whether a tumor is malignant (M) or benign (B) based on various features of cell nuclei. The dataset is the Wisconsin Breast Cancer Dataset, which includes multiple features such as radius, texture, smoothness, and others.

## Project Overview

The goal of this project is to predict cancer diagnosis (Malignant or Benign) using Logistic Regression. The model is trained using the features from the dataset, and the performance is evaluated based on accuracy, precision, recall, and F1-score.

## Dataset

The dataset is available from the UCI Machine Learning Repository. It contains 569 samples, each with 32 features (excluding the ID column). The features represent measurements of cell nuclei from breast cancer biopsies. The target variable is `diagnosis`, which has two possible values:
- `M` for Malignant
- `B` for Benign

The dataset contains the following columns:
- `id`: Unique identifier for each sample
- `diagnosis`: The diagnosis of the tumor (Malignant or Benign)
- `radius_mean`, `texture_mean`, etc.: Various feature measurements

## Libraries Used

- `pandas`: Data manipulation and analysis
- `sklearn`: Machine learning and model evaluation
- `matplotlib`: Data visualization (optional)

## Steps Involved

1. **Data Loading**: The dataset is loaded from a CSV file.
2. **Data Preprocessing**: The `id`, `diagnosis`, and `Unnamed: 32` columns are dropped, and the target variable (`diagnosis`) is separated from the feature set.
3. **Train-Test Split**: The dataset is split into training and testing sets (80% training, 20% testing).
4. **Model Training**: A Logistic Regression model is trained on the training set.
5. **Model Evaluation**: The model's performance is evaluated using the test set, and the results are displayed using metrics like accuracy, precision, recall, and F1-score.

## Code

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Load dataset
cancer = pd.read_csv('https://github.com/YBIFoundation/Dataset/raw/main/Cancer.csv')

# Preprocessing
y = cancer['diagnosis']
X = cancer.drop(['id','diagnosis','Unnamed: 32'],axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=2529)

# Train model
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Confusion Matrix:")
print(conf_matrix)
print("\nAccuracy Score:", accuracy)
print("\nClassification Report:")
print(class_report)


**Results**
Accuracy: 96.49%
Precision: 96% (Malignant), 97% (Benign)
Recall: 96% (Malignant), 97% (Benign)
F1-Score: 96% (Malignant), 97% (Benign)


**Confusion Matrix**
```lua
[[64,  2]
 [ 2, 46]]

**Conclusion**
The Logistic Regression model performs very well with an accuracy of 96.49%. The model is able to predict whether a tumor is malignant or benign with high precision and recall.
