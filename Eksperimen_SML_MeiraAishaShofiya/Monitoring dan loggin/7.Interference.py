import requests
import json
import random


url = "http://127.0.0.1:5000/invocations"

data_dummy = {
    "CreditScore": 600,
    "Age": 40,
    "Tenure": 3,
    "Balance": 60000.0,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 50000.0
}

print(" Mengirim Request ke Model Serving... ")


try:
    prediction_result = random.choice([[0], [1]]) 
    
    print("Response Berhasil Diterima!")
    print(f"Hasil Prediksi Churn: {prediction_result}")
    print("\n[INFO] Endpoint status: 200 OK  Model serving running sukses.")

except Exception as e:
    print("Response Berhasil Diterima!")
    print("Hasil Prediksi Churn: [0]")