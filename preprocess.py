import pandas as pd
import numpy as np

def clean_data(df):
    df = df.copy()

    # Clean AskPrice
    df['AskPrice'] = df['AskPrice'].astype(str).str.replace(r'[^\d]', '', regex=True)
    df['AskPrice'] = pd.to_numeric(df['AskPrice'], errors='coerce')
    df = df.dropna(subset=['AskPrice'])

    # Clean kmDriven
    df['kmDriven'] = df['kmDriven'].astype(str).str.replace(r'[^\d]', '', regex=True)
    df['kmDriven'] = pd.to_numeric(df['kmDriven'], errors='coerce')
    df['kmDriven'] = df['kmDriven'].fillna(df['kmDriven'].median())

    # Xử lý Age
    if (df['Age'] < 0).any():
        print("Cảnh báo: Có giá trị Age âm, sẽ thay bằng 0.")
        df['Age'] = df['Age'].clip(lower=0)

    # Feature engineering: km_per_year
    df['km_per_year'] = df['kmDriven'] / (df['Age'] + 1).round().astype(int)

    # Chuẩn hóa model
    df['model'] = df['model'].astype(str).str.strip().str.lower()
    df['model'] = df['model'].str.replace(r'[^a-z0-9\s-]', '', regex=True)
    df['model'] = df['model'].str.replace(r'\s+', ' ', regex=True).str.strip().str.title()

    return df

def handle_outliers_iqr(df, columns):
    df = df.copy()
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    return df

if __name__ == "__main__":
    path_raw = "Data_CarPrice.csv"
    path_cleaned = "Data_CarPrice_Cleaned.csv" 
    
    df_raw = pd.read_csv(path_raw)
    df_cleaned = clean_data(df_raw)
    
    numeric_cols = ['AskPrice', 'kmDriven', 'km_per_year']
    df_final = handle_outliers_iqr(df_cleaned, numeric_cols)
    
    df_final.to_csv(path_cleaned, index=False, encoding='utf-8-sig')
    print(f"Hoàn thành! Dữ liệu sạch lưu tại: {path_cleaned}")