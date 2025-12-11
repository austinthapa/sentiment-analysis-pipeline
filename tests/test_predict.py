# from fastapi.testclient import TestClient
# from app import app

# client = TestClient(app)

# def test_health():
#     response = client.get("/health")
#     assert response.status_code == 200
#     assert response.json() == {"status": "Okay"}
    
# def test_predict():
#     response = client.post("/predict", json = {"text": "I love this product"})
#     assert response.status_code == 200
#     assert "prediction" in response.json()