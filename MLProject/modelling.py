import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":

    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_dir, "churn_banking_preprocessing", "train_clean.csv")
    test_path = os.path.join(base_dir, "churn_banking_preprocessing", "test_clean.csv")

    try:
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
    except FileNotFoundError:
        print("Error: File dataset tidak ditemukan.")
        exit()

    X_train = train_data.drop(columns=['Exited'])
    y_train = train_data['Exited']
    X_test = test_data.drop(columns=['Exited'])
    y_test = test_data['Exited']

    with mlflow.start_run(run_name="RandomForest_Baseline"):
        model = RandomForestClassifier(random_state=42, n_estimators=50, max_depth=10)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Accuracy: {accuracy}")
        
        mlflow.log_param("n_estimators", 50)
        mlflow.log_param("max_depth", 10)
        mlflow.log_param("random_state", 42)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")
        
        print("\nModel dan metrik berhasil di-log ke MLflow.")
