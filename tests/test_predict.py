import os, sys

project_root = os.path.abspath(os.path.join(os.path.abspath(__file__), "..", ".."))
sys.path.insert(0, project_root)

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "Okay"}
    
def test_predict():
    response = client.post("/predict", json = {"text": "I love this product"})
    assert response.status_code == 200
    assert "prediction" in response.json()