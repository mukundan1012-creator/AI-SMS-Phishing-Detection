import streamlit as st
import pickle
import re
import string
import os
import pandas as pd
from scipy.sparse import hstack

current_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(current_dir, "..", "Models", "spam_model.pkl")
tfidf_path = os.path.join(current_dir, "..", "Models", "tfidf_vectorizer.pkl")

with open(model_path, "rb") as file:
    model = pickle.load(file)

with open(tfidf_path, "rb") as file:
    tfidf = pickle.load(file)

# ==========================
# SIDEBAR
# ==========================

st.sidebar.header("📋 Project Information")

st.sidebar.write("### Model")
st.sidebar.write("Logistic Regression (Balanced, C=5)")

st.sidebar.write("### Performance")
st.sidebar.write("🏆 Accuracy : 99.01%")
st.sidebar.write("🎯 ROC-AUC : 0.9959")
st.sidebar.write("🎯 Recall (Spam) : 95%")
st.sidebar.write("🚨 False Negatives : 8 / 149")
st.sidebar.write("⚠️ False Positives : 3 / 966")

st.sidebar.write("### Features Used")
st.sidebar.write("• TF-IDF")
st.sidebar.write("• Message Length")
st.sidebar.write("• Digit Count")
st.sidebar.write("• Special Character Count")
st.sidebar.write("• URL Count")

st.sidebar.markdown("---")
st.sidebar.caption(
    "⚠️ 8 out of 149 real spam messages are missed (false negatives) — "
    "see README for full confusion matrix and evaluation."
)

# ==========================
# MAIN PAGE
# ==========================

st.title("📱 AI Powered SMS Phishing Detection")

st.write(
    "Detect whether an SMS message is Ham or Spam using Machine Learning."
)
with st.expander("🔍 How It Works"):
    st.markdown("""
        SMS Message  
        ↓  
        TF-IDF Vectorization  
        ↓  
        Feature Engineering  
        ↓  
        Logistic Regression  
        ↓  
        Spam / Ham Prediction
        """)
# ==========================
# USER INPUT
# ==========================

user_message = st.text_area("Enter SMS Message")

# ==========================
# PREDICT BUTTON
# ==========================

if st.button("Predict"):

    if user_message.strip() == "":
        st.warning("Please enter a message.")
    else:
        message_tfidf = tfidf.transform([user_message])

        message_length = len(user_message)
        digit_count = sum(char.isdigit() for char in user_message)
        special_char_count = sum(char in string.punctuation for char in user_message)
        url_count = len(re.findall(r'http\S+|www\S+', user_message))

        extra_features = pd.DataFrame(
            [[message_length, digit_count, special_char_count, url_count]],
            columns=["message_length", "digit_count", "special_char_count", "url_count"]
        )

        message_combined = hstack((message_tfidf, extra_features))

        prediction = model.predict(message_combined)
        probability = model.predict_proba(message_combined)
        spam_probability = probability[0][1]

        st.subheader("Prediction Result")

        if prediction[0] == 1:
            st.error("🚨 SPAM MESSAGE DETECTED")
        else:
            st.success("✅ HAM MESSAGE DETECTED")

        if spam_probability >= 0.50:
            st.error(f"🔴 Spam Probability : {spam_probability:.2%}")
        else:
            st.success(f"🟢 Spam Probability : {spam_probability:.2%}")

        st.subheader("Feature Analysis")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("📏 Message Length", message_length)
            st.metric("🔢 Digit Count", digit_count)
        with col2:
            st.metric("❗ Special Characters", special_char_count)
            st.metric("🔗 URL Count", url_count)

        st.caption(
            "Note: This model can occasionally miss real spam messages (false negatives) "
            "or flag legitimate messages as spam (false positives). See README for full metrics."
        )

st.subheader("🧪 Example Messages")

st.write("HAM Example")
st.code("Hey bro, where are you? We are waiting near the college gate.")

st.write("SPAM Example")
st.code("Congratulations! You have won Rs 50000 cash. Click http://claim-now.com to collect your reward immediately.")

st.markdown("---")
st.caption("Developed by Mukundan D | B.E Electronics and Communication Engineering")