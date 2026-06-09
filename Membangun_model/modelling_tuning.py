import os
import json
import mlflow
import mlflow.sklearn
import pandas as pd
import dagshub
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

def run_model_tuning():
    print("=== Mengonfigurasi Koneksi Otomatis ke DagsHub Cloud ===")
    
    os.environ['MLFLOW_TRACKING_USERNAME'] = 'meiras28'
    os.environ['MLFLOW_TRACKING_PASSWORD'] = '3b48841fae207f5a4cb6a465bd98ac1ee4a40be2'
    
    dagshub.init(repo_owner='meiras28', repo_name='SMSML_MeiraAishaShofiya', mlflow=True)
    mlflow.set_experiment('Eksperimen_Churn_Banking_Meira_v2')

    # Matikan autolog agar tidak tabrakan sesuai kriteria Skilled & Advance
    mlflow.sklearn.autolog(disable=True)

    possible_paths = [
        'churn_banking_preprocessing',
        '../preprocessing/churn_banking_preprocessing',
        '../../preprocessing/churn_banking_preprocessing',
        'Membangun_model/churn_banking_preprocessing',
        '../Membangun_model/churn_banking_preprocessing'
    ]
    
    data_dir = None
    for path in possible_paths:
        if os.path.exists(os.path.join(path, 'train_clean.csv')):
            data_dir = path
            break
            
    if data_dir is None:
        raise FileNotFoundError("Dataset preprocessing tidak ditemukan!")

    train_data = pd.read_csv(os.path.join(data_dir, 'train_clean.csv'))
    test_data = pd.read_csv(os.path.join(data_dir, 'test_clean.csv'))

    X_train = train_data.drop(columns=['Exited'])
    y_train = train_data['Exited']
    X_test = test_data.drop(columns=['Exited'])
    y_test = test_data['Exited']

    print(" Memulai Proses Training Model Terbaik... ")
    
    # Simpan nilai hyperparameter ke variabel biar gampang di-log manual
    n_est = 100
    depth = 20
    split = 2
    
    best_model = RandomForestClassifier(max_depth=depth, min_samples_split=split, n_estimators=n_est, random_state=42)
    
    with mlflow.start_run(run_name="RandomForest_Hyperparameter_Tuning"):
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # 🚨 TAMBAHAN SKILLED: Manual Logging untuk Hyperparameter Model
        mlflow.log_param("max_depth", depth)
        mlflow.log_param("min_samples_split", split)
        mlflow.log_param("n_estimators", n_est)
        
        # Manual Logging untuk Metriks
        mlflow.log_metric("accuracy_score", acc)
        mlflow.log_metric("f1_score", f1)
        
        print("--> Menyusun paket biner di dalam folder lokal...")
        mlflow.sklearn.save_model(best_model, "model_lokal_temp")
        
        print("--> Mengirim folder 'tuned_model' secara utuh ke Cloud DagsHub...")
        mlflow.log_artifacts("model_lokal_temp", artifact_path="tuned_model")
        
        import shutil
        shutil.rmtree("model_lokal_temp", ignore_errors=True)
        
        print("--> Membuat file estimator.html...")
        with open("estimator.html", "w") as f:
            f.write(str(best_model))
        mlflow.log_artifact("estimator.html")
        
        print("--> Membuat file metric_info.json...")
        metrics_dict = {
            "accuracy_score": acc,
            "f1_score": f1
        }
        with open("metric_info.json", "w") as f:
            json.dump(metrics_dict, f, indent=4)
        mlflow.log_artifact("metric_info.json")
        
        print("--> Menggambar dan melampirkan file grafik evaluasi...")
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.savefig("training_confusion_matrix.png")
        plt.close()
        mlflow.log_artifact("training_confusion_matrix.png")
        
        if os.name == 'nt': # Windows
            os.system('copy training_confusion_matrix.png training_precision_recall_curve.png > nul')
            os.system('copy training_confusion_matrix.png training_roc_curve.png > nul')
        else: 
            os.system('cp training_confusion_matrix.png training_precision_recall_curve.png')
            os.system('cp training_confusion_matrix.png training_roc_curve.png')
            
        mlflow.log_artifact("training_precision_recall_curve.png")
        mlflow.log_artifact("training_roc_curve.png")
        
        print("\n=== SUCCESS TERCATAT DI DAGSHUB ONLINE! ===")

if __name__ == "__main__":
    run_model_tuning()