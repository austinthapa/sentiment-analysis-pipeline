import os
import yaml
import joblib

from pydantic import BaseModel
from fastapi import FastAPI

# Configure model path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_config():
    with open(os.path.join(BASE_DIR, "config.yaml"), "r") as file:
        return yaml.safe_load(file)
    
config = load_config()
MODEL_PATH = config["MODEL_PATH"]
VECTORIZER_PATH = config["VECTORIZER_PATH"]


# --- Initialize the app ---
app = FastAPI(
    title="Text Classification API",
    description="Predict text sentiment using TF-IDF Vectorizer + LightGBM Model",
    version="1.0"
)

# --- Load the vectorizer and model ---
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# --- Define the Input Schema
class TextInput(BaseModel):
    text: str
    
# --- Define the routes ---
@app.get("/")
def home():
    return {
        "message": "Welcome to the Text Classification API"
    }
    
@app.post("/predict")
def predict(data: TextInput):
    """
    Send text input for prediction.
    """
    X_vec = vectorizer.transform([data.text])
    
    y_pred = model.predict(X_vec)
    
    return {
        "prediction": str(y_pred[0])
    }
    
@app.get("/health")
def health_check():
    return {
        "status": "Okay"
    }