import pandas as pd
import numpy as np


# Chia train/test (80/20)
def train_test_split(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    idx = np.random.permutation(len(y))
    split = int(len(y) * (1 - test_size))
    train_idx, test_idx = idx[:split], idx[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

# Chuẩn hoá
class StandardScaler:
    def fit(self, X):
        X = np.array(X, dtype=np.float64)  
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std == 0] = 1

    def transform(self, X):
        X = np.array(X, dtype=np.float64)  
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

def soft_threshold(rho, alpha_l1):

    if rho > alpha_l1:
        return rho - alpha_l1
    elif rho < -alpha_l1:
        return rho + alpha_l1
    else:
        return 0.0

class ElasticNetCD:
    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol

        # sẽ được gán sau khi fit
        self.w = None
        self.b = None

    def compute_loss(self, X, y):
        n = len(y)
        y_pred = X @ self.w + self.b

        mse = np.mean((y - y_pred)**2)
        l1 = np.sum(np.abs(self.w))
        l2 = np.sum(self.w**2)

        return mse + self.alpha * (self.l1_ratio * l1 + 0.5 * (1 - self.l1_ratio) * l2)

    def soft_threshold(self, rho, alpha_l1):
        if rho > alpha_l1:
            return rho - alpha_l1
        elif rho < -alpha_l1:
            return rho + alpha_l1
        else:
            return 0.0

    def fit(self, X, y):
        n, p = X.shape
        self.w = np.zeros(p)
        self.b = 0.0

        alpha_l1 = self.alpha * self.l1_ratio
        alpha_l2 = self.alpha * (1 - self.l1_ratio)

        for iteration in range(self.max_iter):
            w_old = self.w.copy()

            # update bias
            y_pred = X @ self.w + self.b
            self.b = self.b + np.mean(y - y_pred)

            # update từng w_j
            y_pred = X @ self.w + self.b
            for j in range(p):
                rho = (X[:, j] @ (y - y_pred + X[:, j] * self.w[j])) / n
                z_j = (X[:, j] @ X[:, j]) / n

                new_w_j = self.soft_threshold(rho, alpha_l1) / (z_j + alpha_l2)

                y_pred = y_pred + X[:, j] * (new_w_j - self.w[j])
                self.w[j] = new_w_j

            print(f"  Loop: {iteration+1} - Loss: {self.compute_loss(X, y):.4f}")

            if np.max(np.abs(self.w - w_old)) < self.tol:
                print(f"  Hội tụ sau {iteration+1} vòng lặp")
                break

        return self

    
    def predict(self, X):
        return X @ self.w + self.b

    def mean_squared_error(self, y_true, y_pred):
        return np.mean((y_true - y_pred)**2)

    def mean_absolute_error(self, y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    def r2_score(self, y_true, y_pred):
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - y_true.mean())**2)
        return 1 - ss_res / ss_tot

