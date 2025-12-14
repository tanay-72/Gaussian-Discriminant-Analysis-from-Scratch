# Gaussian Discriminant Analysis from Scratch

This project implements **Gaussian Discriminant Analysis (LDA)** from scratch using NumPy, without relying on sklearn models.

## Dataset
- Breast Cancer Wisconsin (Diagnostic)
- Binary classification (Benign / Malignant)

## What was implemented
- Maximum Likelihood Estimation of:
  - Class priors
  - Class means
  - Shared covariance matrix
- Feature scaling using StandardScaler
- MAP decision rule for prediction
- Covariance regularization for numerical stability

## Results
- Accuracy: ~95%
- Performance comparable to sklearn's LDA

## Concepts Used
- Generative models
- Gaussian likelihood
- Exponential family
- Linear decision boundary
- Numerical stability in matrix inversion

