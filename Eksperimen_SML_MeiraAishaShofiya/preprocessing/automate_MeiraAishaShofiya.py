import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def run_preprocessing_pipeline():
    possible_paths = [
        "Churn_Modelling.csv",                                      
        "../Churn_Modelling.csv",                               
        "../../Churn_Modelling.csv",                              
        "Eksperimen_SML_MeiraAishaShofiya/Churn_Modelling.csv"       
    ]
    
    input_path = None
    for path in possible_paths:
        if os.path.exists(path):
            input_path = path
            break
            
    if input_path is None:
        raise FileNotFoundError("Dataset 'Churn_Modelling.csv' tidak ditemukan di folder mana pun.")
        
    print(f"=== Menggunakan Dataset dari Path: {input_path} ===")
    df_churn = pd.read_csv(input_path)

    X = df_churn.drop(columns=['RowNumber', 'CustomerId', 'Surname', 'Exited'])
    y = df_churn['Exited']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Scaling data numerik
    num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    scaler = StandardScaler()
    X_train_num = pd.DataFrame(scaler.fit_transform(X_train[num_cols]), columns=num_cols, index=X_train.index)
    X_test_num = pd.DataFrame(scaler.transform(X_test[num_cols]), columns=num_cols, index=X_test.index)


    cat_cols = ['Geography', 'Gender']
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_train_cat = pd.DataFrame(encoder.fit_transform(X_train[cat_cols]), columns=encoder.get_feature_names_out(cat_cols), index=X_train.index)
    X_test_cat = pd.DataFrame(encoder.transform(X_test[cat_cols]), columns=encoder.get_feature_names_out(cat_cols), index=X_test.index)

    passthrough_cols = ['HasCrCard', 'IsActiveMember']

    train_clean = pd.concat([X_train_num, X_train[passthrough_cols], X_train_cat, y_train], axis=1)
    test_clean = pd.concat([X_test_num, X_test[passthrough_cols], X_test_cat, y_test], axis=1)

    output_dir = 'churn_banking_preprocessing'
    
    # Jika dijalankan dari dalam folder processing, sesuaikan output_dir agar keluar ke folder utama
    if os.path.basename(os.getcwd()) == "processing":
        output_dir = os.path.join("..", "churn_banking_preprocessing")
        
    os.makedirs(output_dir, exist_ok=True)
    train_clean.to_csv(os.path.join(output_dir, 'train_clean.csv'), index=False)
    test_clean.to_csv(os.path.join(output_dir, 'test_clean.csv'), index=False)
    
    print("=== Preprocessing Automate sukses ===")
    print(f"Data latihan berhasil disimpan di folder: {output_dir}")
    print(f"Ukuran tabel hasil akhir: {train_clean.shape}")

if __name__ == "__main__":
    run_preprocessing_pipeline()