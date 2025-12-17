import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):
    # Hapus data duplikat
    if df.duplicated().sum() > 0:
        df = df.drop_duplicates()
    
    # Label Encoding
    le = LabelEncoder()
    object_cols = df.select_dtypes(include=['object']).columns
    for col in object_cols:
        df[col] = le.fit_transform(df[col])
        
    # Scaling Age
    scaler = StandardScaler()
    df['Age'] = scaler.fit_transform(df[['Age']])
    
    return df

def save_data(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")

if __name__ == "__main__":
    input_file = "../diabetes_data.csv" 
    output_file = "preprocessing/diabetes_data_preprocessed.csv"
    
    if not os.path.exists('preprocessing'):
        os.makedirs('preprocessing')

    print("Starting preprocessing...")
    df = load_data(input_file)
    df_clean = preprocess_data(df)
    save_data(df_clean, output_file)
    print("Preprocessing completed.")