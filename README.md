---

# 📉 Elastic Net Regression (From Scratch)

## 📌 Overview

This project is a **from-scratch implementation of Elastic Net Regression**, focusing on the core mathematics and optimization process behind the model.

Instead of relying on machine learning libraries, the goal is to:

* Understand how **regularization works**
* Explore how **L1 and L2 penalties interact**
* Implement and analyze the **Coordinate Descent algorithm**

---

## 🧠 Core Concepts

### Elastic Net Regularization

Elastic Net combines two types of regularization:

* **L1 (Lasso)**

  * Encourages sparsity
  * Can shrink some weights exactly to 0 → feature selection

* **L2 (Ridge)**

  * Penalizes large weights
  * Improves stability and reduces variance

Elastic Net balances both using a mixing parameter.

---

### Loss Function

```math
\text{Loss} = \frac{1}{n} \sum (y - \hat{y})^2 
+ \alpha \left( 
\lambda \sum |w_j| 
+ \frac{1 - \lambda}{2} \sum w_j^2 
\right)
```

### Where:

* **α (alpha)** = overall regularization strength
* **λ (l1_ratio)** = balance between L1 and L2
* **w_j** = model coefficients

---

## 🔄 Coordinate Descent Optimization

Instead of updating all parameters at once, Coordinate Descent:

* Updates one weight at a time
* Keeps other weights fixed
* Iterates until convergence

### Key Idea

For each feature `j`:

* Compute partial residual
* Apply soft-thresholding
* Update weight `w_j`

---

## ✂️ Soft Thresholding

```math
S(\rho, \alpha) =
\begin{cases}
\rho - \alpha & \text{if } \rho > \alpha \\
\rho + \alpha & \text{if } \rho < -\alpha \\
0 & \text{otherwise}
\end{cases}
```

This is the key mechanism that:

* Shrinks coefficients
* Forces small values to become exactly 0 (L1 effect)

---

## ⚙️ Implementation Highlights

* Pure **NumPy-based implementation**
* No use of **Scikit-learn** for model training
* Iterative optimization with convergence check:

  * Stops when weight updates are smaller than a tolerance threshold
* Separate handling of:

  * Bias term
  * Regularization components

---

## 🔁 Training Process

1. Initialize weights = 0
2. Repeat until convergence:

   * Update bias term
   * Loop through each feature:

     * Compute `ρ`
     * Apply soft-thresholding
     * Update weight
3. Monitor loss during training

---

## 📊 Evaluation Metrics

* Mean Squared Error (MSE)
* Mean Absolute Error (MAE)
* R² Score

---

## 🎯 Learning Focus

This project emphasizes:

* The mathematical foundation of Elastic Net
* The behavior of regularization terms
* The mechanics of Coordinate Descent optimization
* How models are trained step-by-step internally

---

## 🛠️ Tech Stack

* Python
* NumPy

---

## 📎 Notes

* This implementation is designed for learning and clarity, not production use.
* Understanding this model makes it easier to work with advanced ML libraries later.

---
