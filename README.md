# Digits Classification

This project demonstrates the use of machine learning to classify handwritten digits using two models: Logistic Regression and Random Forest Classifier. The dataset used is the famous `digits` dataset from Scikit-learn.

## Project Overview

The objective of this project is to:
- Train and evaluate two machine learning models: Logistic Regression and Random Forest Classifier.
- Compare their performance based on accuracy and classification metrics.
- Visualize a sample of handwritten digit images with their corresponding labels.

## How to run
Clone or download the repository.
Install the required libraries using:
pip install matplotlib scikit-learn
Run the Python script to see the results and visualizations.

## Requirements

- Python 3.x
- Libraries:
  - `matplotlib`
  - `scikit-learn`
  
To install the required libraries, use:
```bash
pip install matplotlib scikit-learn
## Dataset
The dataset used in this project is the digits dataset from Scikit-learn, which consists of:

Data: 1797 8x8 pixel images of handwritten digits, flattened into 64 feature values.
Target: 1797 labels corresponding to digits 0-9.
## Implementation
1. Loading and Splitting the Dataset
The dataset is loaded using load_digits from Scikit-learn, and then split into training and testing sets using train_test_split.
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

2. Model Training
Logistic Regression
We train the Logistic Regression model using the LogisticRegression class. The max_iter=10000 ensures convergence.
from sklearn.linear_model import LogisticRegression

logistic_model = LogisticRegression(max_iter=10000, random_state=42)
logistic_model.fit(X_train, y_train)
logistic_predictions = logistic_model.predict(X_test)

Random Forest Classifier
We train the Random Forest model using the RandomForestClassifier class with 100 estimators.

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

3. Model Evaluation
Both models are evaluated based on accuracy and classification reports.

from sklearn.metrics import accuracy_score, classification_report

# Logistic Regression Evaluation
print("Logistic Regression Model:")
print(f"Accuracy: {accuracy_score(y_test, logistic_predictions)}")
print("Classification Report:")
print(classification_report(y_test, logistic_predictions))

# Random Forest Evaluation
print("Random Forest Model:")
print(f"Accuracy: {accuracy_score(y_test, rf_predictions)}")
print("Classification Report:")
print(classification_report(y_test, rf_predictions))

4. Visualizing Sample Digits
We visualize the first 10 digits in the dataset along with their labels.

import matplotlib.pyplot as plt

num_images = 10
fig, axes = plt.subplots(2, 5, figsize=(10, 5), dpi=120)
axes = axes.ravel()

for digit in range(num_images):
    idx = (digits.target == digit).argmax()
    axes[digit].imshow(digits.images[idx], cmap='gray', interpolation='nearest')
    axes[digit].set_title(f"Label: {digit}")
    axes[digit].axis('off')

plt.tight_layout()
plt.show()


