import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
from data_preprocessing import load_data, add_features

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def train_and_save():
    data_path = os.path.join(BASE_DIR, "..", "Dataset", "SMSSpamCollection")
    model_path = os.path.join(BASE_DIR, "..", "Models", "spam_model.pkl")
    tfidf_path = os.path.join(BASE_DIR, "..", "Models", "tfidf_vectorizer.pkl")

    df = load_data(data_path)
    df = add_features(df)

    X = df["message"]
    y = df["label_num"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    tfidf = TfidfVectorizer()
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    extra_cols = ["message_length", "digit_count", "special_char_count", "url_count"]

    X_train_combined = hstack((X_train_tfidf, df.loc[X_train.index, extra_cols]))
    X_test_combined = hstack((X_test_tfidf, df.loc[X_test.index, extra_cols]))

    model = LogisticRegression(max_iter=1000, class_weight="balanced", C=5)
    model.fit(X_train_combined, y_train)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(tfidf_path, "wb") as f:
        pickle.dump(tfidf, f)

    return model, tfidf, X_test_combined, y_test


if __name__ == "__main__":
    train_and_save()