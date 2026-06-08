import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import dagshub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def main():
    repo_owner = "meiras28"
    repo_name = "SMSML_MeiraAishaShofiya"
    
    print("=== Menghubungkan ke DagsHub Remote MLflow untuk Tuning ===")
    os.environ["MLFLOW_TRACKING_USERNAME"] = repo_owner
    
    dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
    mlflow.set_experiment("Banking_Churn_Tuning")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_dir, "churn_banking_preprocessing", "train_clean.csv")
    test_path = os.path.join(base_dir, "churn_banking_preprocessing", "test_clean.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"File tidak ditemukan! Pastikan file ada di: {train_path}")

    print(f"Memuat data dari: {train_path}")
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    X_train = train_data.drop(columns=['Exited'])
    y_train = train_data['Exited']
    X_test = test_data.drop(columns=['Exited'])
    y_test = test_data['Exited']

    print("Memulai proses Hyperparameter Tuning (GridSearchCV)...")
    base_rf = RandomForestClassifier(random_state=42)
    
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [10, 20]
    }
    
    grid_search = GridSearchCV(estimator=base_rf, param_grid=param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    with mlflow.start_run(run_name="RandomForest_Tuned_Manual"):
        print("Mencatat parameter dan metrik secara manual ke DagsHub...")
        
        for param_name, param_val in best_params.items():
            mlflow.log_param(param_name, param_val)
        mlflow.log_param("model_type", "RandomForestClassifier")

        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        cax = ax.matshow(cm, cmap=plt.cm.Blues)
        fig.colorbar(cax)
        
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), va='center', ha='center', fontsize=12, color='black')
                
        ax.set_xticklabels([''] + ['Stay', 'Churn'])
        ax.set_yticklabels([''] + ['Stay', 'Churn'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix', pad=20)
        
        cm_path = "training_confusion_matrix.png"
        plt.savefig(cm_path, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(cm_path)

        fig, ax = plt.subplots(figsize=(8, 6))
        importances = best_model.feature_importances_
        indices = np.argsort(importances)
        ax.barh(range(len(indices)), importances[indices], color='skyblue', align='center')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels(X_train.columns[indices])
        plt.title("Feature Importances")
        plt.xlabel("Relative Importance")
        
        feat_path = "feature_importance.png"
        plt.savefig(feat_path, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(feat_path)

        mlflow.sklearn.log_model(best_model, "model")
        
        if os.path.exists(cm_path): os.remove(cm_path)
        if os.path.exists(feat_path): os.remove(feat_path)
        
        print(f"=== Tuning Sukses! Akurasi Akhir: {acc:.4f} ===")

if __name__ == "__main__":
    main()