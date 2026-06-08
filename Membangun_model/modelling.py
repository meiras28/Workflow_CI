import os
import pandas as pd
import mlflow
import dagshub
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def main():
    # 1. Konfigurasi autentikasi DagsHub
    repo_owner = "meiras28"
    repo_name = "SMSML_MeiraAishaShofiya"
    
    print("=== Menghubungkan ke DagsHub Remote MLflow ===")
    os.environ["MLFLOW_TRACKING_USERNAME"] = repo_owner
    
    dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
    mlflow.set_experiment("Banking_Churn_Baseline")

    # 2. Menyesuaikan folder ke 'churn_banking_preprocessing' sesuai sidebar VS Code
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_dir, "churn_banking_preprocessing", "train_clean.csv")
    test_path = os.path.join(base_dir, "churn_banking_preprocessing", "test_clean.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"File tidak ditemukan! Pastikan file ada di: {train_path}")

    print(f"Memuat data dari: {train_path}")
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # Pisahkan Fitur dan Target
    X_train = train_data.drop(columns=['Exited'])
    y_train = train_data['Exited']
    X_test = test_data.drop(columns=['Exited'])
    y_test = test_data['Exited']

    # 3. Aktifkan Autolog
    mlflow.autolog(log_models=True)
    
    with mlflow.start_run(run_name="RandomForest_Baseline"):
        print("Melatih Baseline Model (Random Forest)...")
        model = RandomForestClassifier(random_state=42, n_estimators=50, max_depth=10)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"=== Baseline Model Sukses! Akurasi: {acc:.4f} ===")

if __name__ == "__main__":
    main()