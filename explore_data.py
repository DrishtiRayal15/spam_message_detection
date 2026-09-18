"""
explore_data.py
----------------
Basic look at the dataset before training anything.
Run: python explore_data.py
"""

import pandas as pd

DATA_FILE = "spam.csv"


def load(path):
    data = pd.read_csv(path, encoding="latin-1")
    data = data.iloc[:, :2]
    data.columns = ["label", "text"]
    data.dropna(inplace=True)
    data["label"] = data["label"].str.lower().str.strip()
    return data


if __name__ == "__main__":
    data = load(DATA_FILE)

    print("Rows, Columns:", data.shape)
    print("\nClass balance:")
    print(data["label"].value_counts())

    data["length"] = data["text"].astype(str).str.len()
    print("\nAverage length by class:")
    print(data.groupby("label")["length"].mean())

    print("\nLongest message:")
    longest = data.loc[data["length"].idxmax()]
    print(f"({longest['label']}) {longest['text'][:120]}...")

    print("\nA few spam examples:")
    for msg in data[data["label"] == "spam"]["text"].head(3):
        print("-", msg[:100])
