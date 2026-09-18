@echo off
echo Installing packages...
pip install -r requirements.txt

if not exist model.pkl (
    echo Training model for the first time...
    python train_model.py
) else (
    echo Found existing model.pkl, skipping training.
)

echo Launching app...
python app.py
pause
