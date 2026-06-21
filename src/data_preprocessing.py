import pandas as pd
import string
import re
import nltk
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

def load_data(path):
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["label", "message"]
    )
    return df

def clean_text(text):
    text=text.lower()

    cleaned_text=""
    for char in text:
        if char not in string.punctuation:
            cleaned_text += char

    tokens = cleaned_text.split()

    filtered_message = []

    for word in tokens:
        if word not in stop_words:
            filtered_message.append(word)

    return filtered_message

def add_features(df):
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
    df["cleaned_message"] = df["message"].apply(clean_text)
    df["message_length"] = df["message"].apply(len)
    df["digit_count"] = df["message"].apply(
        lambda x: sum(char.isdigit() for char in x)
    )
    df["special_char_count"] = df["message"].apply(
        lambda x: sum(char in string.punctuation for char in x)
    )
    df["url_count"] = df["message"].apply(
        lambda x: len(re.findall(r'http\S+|www\S+', x))
    )
    return df


