import re
import pickle
from flask import Flask, request, render_template

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)

with open("model.pkl", "rb") as f:
    clf = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)


def normalize(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@app.route("/", methods=["GET", "POST"])
def home():

    label = None
    text = ""

    if request.method == "POST":

        text = request.form.get("text", "")

        vec = tfidf.transform([
            normalize(text)
        ])

        label = clf.predict(vec)[0]

    return render_template(
        "index.html",
        label=label,
        text=text
    )


@app.route("/predict", methods=["POST"])
def predict():

    payload = request.get_json(force=True)

    text = payload.get("text", "")

    vec = tfidf.transform([
        normalize(text)
    ])

    label = clf.predict(vec)[0]

    return {"label": label}


if __name__ == "__main__":
    app.run(debug=True)
