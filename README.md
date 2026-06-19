# AI Powered SMS Phishing Detection

## Project Overview

This project is a Machine Learning based SMS Phishing Detection System developed using Python, Scikit-learn and Streamlit.

The application classifies SMS messages as either:

* Ham (Legitimate Message)
* Spam (Phishing / Fraudulent Message)

The project uses TF-IDF vectorization, feature engineering and Logistic Regression to detect phishing messages with high accuracy.

---

## Dataset

Dataset Used:

* SMS Spam Collection Dataset
* Contains labeled SMS messages classified as spam or ham

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Pickle
* SciPy

---

## Feature Engineering

In addition to TF-IDF features, the following handcrafted features were added:

* Message Length
* Digit Count
* Special Character Count
* URL Count

These features helped improve spam detection recall.

---

## Machine Learning Models Tested

### 1. Multinomial Naive Bayes

* Accuracy: 97.40%
* Recall: 85%

### 2. Logistic Regression (Selected Model)

* Accuracy: 99.01%
* Recall: 95%

The Logistic Regression model achieved the best performance and was selected as the final model.

---

## Model Improvements

The following improvements were applied:

* Feature Engineering
* Class Weight Balancing
* Hyperparameter Tuning (C = 5)

These improvements reduced False Negatives and improved recall.

---

## Final Results

| Metric          | Value  |
| --------------- | ------ |
| Accuracy        | 99.01% |
| Recall          | 95%    |
| False Negatives | 8      |

---

## Streamlit Application

The project includes a Streamlit web application where users can:

* Enter SMS messages
* Detect Spam or Ham messages
* View Spam Probability
* View Feature Analysis

---

## Project Structure

SMS_Phishing_ProjectFinal

├── App/app.py

├── Dataset/spam.csv

├── Models/spam_model.pkl

├── Models/tfidf_vectorizer.pkl

├── SMS_Phishing_Project.ipynb

├── requirements.txt

└── README.md

---

## Author

Mukundan D

B.E Electronics and Communication Engineering
