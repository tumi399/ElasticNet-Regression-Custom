import pickle
import pandas as pd
import numpy as np

from preprocess import clean_data
from encoding import transform_target_encoder
from elasticnet import predict

# Load model
with open("elastic_model_weights.pkl", "rb") as f:
    model = pickle.load(f)

w           = model["weights"]
b           = model["bias"]
scaler      = model["scaler"]
columns     = model["columns"]
target_maps = model["target_maps"]
global_mean = model["global_mean"]

# Load & Clean
df = pd.read_csv("Data_CarPrice_Prediction.csv")
df = clean_data(df)

y_true = None
if "AskPrice" in df.columns:
    y_true = df["AskPrice"].values
    df = df.drop(columns=["AskPrice"])

# Target Encoding (dùng map từ lúc train)
for col in ['Brand', 'model']:
    if col in df.columns and col in target_maps:
        df[col] = transform_target_encoder(df, col, target_maps[col], global_mean)

# One-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Align columns
X = df.reindex(columns=columns, fill_value=0)
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
X = X.values.astype(np.float64)

# Scale
X = scaler.transform(X)

# Predict
y_pred = predict(X, w, b)

# Hiển thị kết quả
result = pd.DataFrame({"AskPrice_Predicted": y_pred.round(0).astype(int)})

if y_true is not None:
    result.insert(0, "AskPrice_Actual", y_true.astype(int))
    result["Difference"] = result["AskPrice_Predicted"] - result["AskPrice_Actual"]
    result["Error%"] = (result["Difference"] / result["AskPrice_Actual"] * 100).round(2)

print(result.to_string())