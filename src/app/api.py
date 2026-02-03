from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI(title="Titanic Survival API")

# Load trained model once at startup
model = joblib.load("model.joblib")


@app.get("/")
def root():
    return {"message": "Titanic survival prediction API"}


@app.get("/predict")
def predict(
    age: float,
    fare: float,
    sex: str,
    embarked: str,
):
    # Create DataFrame with the same column names as training data
    # The model pipeline expects a DataFrame with columns: Age, Fare, Sex, Embarked
    X = pd.DataFrame({
        "Age": [age],
        "Fare": [fare],
        "Sex": [sex],
        "Embarked": [embarked],
    })
    
    prediction = model.predict(X)[0]
    
    return {"survived": bool(prediction)}