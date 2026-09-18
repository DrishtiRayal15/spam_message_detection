"""
train_model.py
---------------
Builds the spam classifier and saves three things:
    model.pkl           trained classifier
    vectorizer.pkl       fitted text-to-number converter
    model_metrics.json   how well it performed

Run: python train_model.py
"""

import re
import json
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

DATA_FILE = "spam.csv"


def normalize(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_dataset(path):
    data = pd.read_csv(path, encoding="latin-1")
    data = data.iloc[:, :2]
    data.columns = ["label", "text"]
    data.dropna(inplace=True)
    data["label"] = data["label"].str.lower().str.strip()
    data["norm_text"] = data["text"].apply(normalize)
    return data


def main():
    data = load_dataset(DATA_FILE)

    train_x, test_x, train_y, test_y = train_test_split(
        data["norm_text"], data["label"], test_size=0.25, random_state=7, stratify=data["label"]
    )

    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=4000)
    train_vecs = tfidf.fit_transform(train_x)
    test_vecs = tfidf.transform(test_x)

    clf = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)
    clf.fit(train_vecs, train_y)

    preds = clf.predict(test_vecs)

    scorecard = {
        "accuracy": round(accuracy_score(test_y, preds), 4),
        "precision": round(precision_score(test_y, preds, pos_label="spam"), 4),
        "recall": round(recall_score(test_y, preds, pos_label="spam"), 4),
        "f1_score": round(f1_score(test_y, preds, pos_label="spam"), 4),
        "samples_trained": len(train_x),
        "samples_tested": len(test_x),
    }

    print("Results:", scorecard)

    with open("model.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf, f)
    with open("model_metrics.json", "w") as f:
        json.dump(scorecard, f, indent=2)

    print("Saved model.pkl, vectorizer.pkl, model_metrics.json")


if __name__ == "__main__":
    main()
