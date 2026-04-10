# 🚗 Elastic Net Regression from Scratch

## 📌 Overview
This project implements **Elastic Net Regression from scratch** using Python, without relying on machine learning libraries like Scikit-learn.

It covers the full machine learning pipeline:
- Data preprocessing  
- Feature engineering  
- Encoding categorical variables  
- Model training using **Coordinate Descent**  
- Evaluation and prediction  

The goal is to deeply understand how Elastic Net works internally, instead of just using built-in libraries.

---

## ⚙️ Features

### 1. Data Preprocessing
- Cleans raw dataset (removes invalid characters, handles missing values)
- Handles outliers using **IQR method**
- Feature engineering:
  - `km_per_year = kmDriven / (Age + 1)`
- Standardizes text data (e.g., car model names)

### 2. Encoding
- **One-hot encoding** for categorical features:
  - Transmission, Owner, FuelType
- **Target encoding** for high-cardinality features:
  - Brand, Model
- Applies **log transformation** on target variable (`AskPrice`)

### 3. Custom Elastic Net Model
Implemented from scratch with:
- **Coordinate Descent algorithm**
- Combined regularization:
  - L1 (Lasso)
  - L2 (Ridge)
- Soft-thresholding function
- Custom loss function

### 4. Model Evaluation
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R² Score

### 5. Prediction Pipeline
- Loads trained model from `.pkl`
- Applies same preprocessing & encoding
- Generates predictions with error comparison (if ground truth available)

---

## 🧠 How Elastic Net Works

Elastic Net combines L1 and L2 regularization:

- **L1 (Lasso):** Feature selection (drives some weights to 0)
- **L2 (Ridge):** Stabilizes model and reduces variance

### Loss Function

```math
Loss = MSE + α * (λ * L1 + (1 - λ) * L2)
