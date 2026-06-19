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
* NLTK
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

| Model                                 | Accuracy | Recall |
| ------------------------------------- | -------- | ------ |
| Multinomial Naive Bayes               | 97.40%   | 85%    |
| Logistic Regression (Balanced, C = 5) | 99.01%   | 95%    |

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

## Project Structure

```text
SMS_Phishing_ProjectFinal

├── App/
│   └── app.py

├── Dataset/
│   └── SMSSpamCollection

├── Models/
│   ├── spam_model.pkl
│   └── tfidf_vectorizer.pkl

├── SMS_Phishing_ProjectFinal.ipynb

├── requirements.txt

└── README.md
```

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/mukundan1012-creator/AI-SMS-Phishing-Detection.git
```

### 2. Move into the Project Directory

```bash
cd AI-SMS-Phishing-Detection
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application

```bash
streamlit run App/app.py
```

### 5. Open in Browser

The application will automatically open in your browser.

If it does not open automatically, visit:

```text
http://localhost:8501
```

## Author

Mukundan D

B.E Electronics and Communication Engineering
