import streamlit as st
import pickle
import re
import string
import pandas as pd
from scipy.sparse import hstack

import os
import pickle

current_dir = os.path.dirname(__file__)

model_path = os.path.join(
    current_dir,
    "..",
    "Models",
    "spam_model.pkl"
)

tfidf_path = os.path.join(
    current_dir,
    "..",
    "Models",
    "tfidf_vectorizer.pkl"
)

with open(model_path, "rb") as file:
    model = pickle.load(file)

with open(tfidf_path, "rb") as file:
    tfidf = pickle.load(file)
# SIDEBAR

# ==========================

st.sidebar.header("📋 Project Information")

st.sidebar.write("### Model")
st.sidebar.write("Logistic Regression")

st.sidebar.write("### Improvements")
st.sidebar.write("✅ Class Weight Balanced")
st.sidebar.write("✅ C = 5")

st.sidebar.write("### Performance")
st.sidebar.write("🏆 Accuracy : 99.01%")
st.sidebar.write("🎯 Recall : 95%")
st.sidebar.write("🚨 False Negatives : 8")

st.sidebar.write("### Features Used")
st.sidebar.write("• TF-IDF")
st.sidebar.write("• Message Length")
st.sidebar.write("• Digit Count")
st.sidebar.write("• Special Character Count")
st.sidebar.write("• URL Count")

# ==========================

# MAIN PAGE

# ==========================

st.title("📱 AI Powered SMS Phishing Detection")

st.write(
"Detect whether an SMS message is Ham or Spam using Machine Learning."
)

# ==========================
# HOW IT WORKS
# ==========================

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

        # TF-IDF

        message_tfidf = tfidf.transform(
            [user_message]
        )

        # EXTRA FEATURES

        message_length = len(user_message)

        digit_count = sum(
            char.isdigit()
            for char in user_message
        )

        special_char_count = sum(
            char in string.punctuation
            for char in user_message
        )

        url_count = len(
            re.findall(
                r'http\S+|www\S+',
                user_message
            )
        )

        # CREATE DATAFRAME

        extra_features = pd.DataFrame(
            [[
                message_length,
                digit_count,
                special_char_count,
                url_count
            ]],
            columns=[
                "message_length",
                "digit_count",
                "special_char_count",
                "url_count"
            ]
        )

        # COMBINE FEATURES

        message_combined = hstack(
            (
                message_tfidf,
                extra_features
            )
        )

        # PREDICTION

        prediction = model.predict(
            message_combined
        )

        probability = model.predict_proba(
            message_combined
        )

        spam_probability = probability[0][1]

        # RESULT

        st.subheader("Prediction Result")

        if prediction[0] == 1:

            st.error(
                "🚨 SPAM MESSAGE DETECTED"
            )

        else:

            st.success(
                "✅ HAM MESSAGE DETECTED"
            )

        # PROBABILITY

        if spam_probability >= 0.50:

            st.error(
                f"🔴 Spam Probability : {spam_probability:.2%}"
            )

        else:

            st.success(
                f"🟢 Spam Probability : {spam_probability:.2%}"
            )

        # FEATURE ANALYSIS

        st.subheader("Feature Analysis")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "📏 Message Length",
                message_length
            )

            st.metric(
                "🔢 Digit Count",
                digit_count
            )

        with col2:

            st.metric(
                "❗ Special Characters",
                special_char_count
            )

            st.metric(
                "🔗 URL Count",
                url_count
            )
# ==========================
# EXAMPLE MESSAGES
# ==========================

st.subheader("🧪 Example Messages")

st.write("HAM Example")

st.code(
    "Hey bro, where are you? We are waiting near the college gate."
)

st.write("SPAM Example")

st.code(
    "Congratulations! You have won Rs 50000 cash. Click http://claim-now.com to collect your reward immediately."
)
# ==========================
# FOOTER
# ==========================

st.markdown("---")

st.caption(
    "Developed by Mukundan D | B.E Electronics and Communication Engineering"
)