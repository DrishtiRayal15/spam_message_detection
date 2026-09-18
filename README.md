# Spam Checker

Classifies a text message as spam or ham (normal) using TF-IDF + Logistic Regression.

## Files
- `explore_data.py` — look at the dataset before training
- `train_model.py` — trains the model, saves model.pkl / vectorizer.pkl / model_metrics.json
- `app.py` — Flask web app + `/predict` API for live checking
- `model.pkl`, `vectorizer.pkl`, `model_metrics.json` — created after training (not included until you run train_model.py)
- `requirements.txt`, `run.bat` — setup helpers

## How to run
1. Put a labeled dataset as `spam.csv` (columns: label, text) in this folder.
2. `pip install -r requirements.txt`
3. `python train_model.py`
4. `python app.py` → open http://127.0.0.1:5000

Or on Windows just double-click `run.bat`.
