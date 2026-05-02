from fastapi.testclient import TestClient
from src.api import app  # Ganti 'main' dengan nama file tempat 'app = FastAPI()' Anda berada

client = TestClient(app)

def test_predict_endpoint_success():
    # 1. Siapkan Mock Request (Data Palsu) yang sesuai format .NET
    mock_payload = {
        "user_id": "9f55a4ee-7be6-4c54-a5c6-bf173ea2ad74",
        "row": {
            "wpm": 55, "accuracy": 0.93,
            "error_left_pinky": 0.02, "error_left_ring": 0.01,
            "error_left_middle": 0.01, "error_left_index": 0.01,
            "error_right_index": 0.02, "error_right_middle": 0.01,
            "error_right_ring": 0.01, "error_right_pinky": 0.01,
            "dwell_left_pinky": 115, "dwell_left_ring": 98,
            "dwell_left_middle": 91, "dwell_left_index": 86,
            "dwell_right_index": 88, "dwell_right_middle": 93,
            "dwell_right_ring": 99, "dwell_right_pinky": 121,
            "flight_left_pinky": 220, "flight_left_ring": 198,
            "flight_left_middle": 180, "flight_left_index": 171,
            "flight_right_index": 174, "flight_right_middle": 182,
            "flight_right_ring": 203, "flight_right_pinky": 231
        }
    }

    # 2. Hit endpoint /predict
    response = client.post("/predict", json=mock_payload)

    # 3. Assertions (Validasi Hasil)
    assert response.status_code == 200 # Pastikan HTTP 200 OK

    data = response.json()
    assert "prediction" in data # Pastikan ada kunci 'prediction'
    assert "confidence" in data
    assert "is_fallback_used" in data
    assert type(data["confidence"]) == float # Pastikan tipe datanya angka
    assert type(data["is_fallback_used"]) == bool # Pastikan tipe datanya boolean
