# Linear Regression from Scratch: OLS, Gradient Descent, Regularization, and Momentum

This repository implements linear regression completely from scratch. It covers both the analytical solution via Ordinary Least Squares (OLS) and an iterative solution using Gradient Descent. The implementation also incorporates ridge regularization (L2 regularization) and momentum to improve convergence. This README provides a mathematically rigorous explanation of each component and discusses why these techniques are useful.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Ordinary Least Squares (OLS)](#ordinary-least-squares-ols)
   - [Mathematical Formulation](#mathematical-formulation)
   - [Derivation of the Normal Equations](#derivation-of-the-normal-equations)
3. [Gradient Descent Optimization](#gradient-descent-optimization)
   - [Standard Gradient Descent](#standard-gradient-descent)
   - [Ridge Regularization](#ridge-regularization)
   - [Momentum in Gradient Descent](#momentum-in-gradient-descent)
4. [Implementation Details](#implementation-details)
5. [Usage](#usage)
6. [Conclusion](#conclusion)

---

## Introduction

Linear regression is one of the simplest and most widely used models in statistics and machine learning. The goal is to fit a linear model to data by minimizing the error between the predicted and observed values. There are two main approaches:

- **Analytical Solution (OLS):** Computes the model parameters in closed form.
- **Iterative Optimization (Gradient Descent):** Approximates the solution through iterative updates.

Enhancements such as **ridge regularization** help prevent overfitting, and **momentum** can speed up and stabilize the convergence of gradient descent.

---

## Ordinary Least Squares (OLS)

### Mathematical Formulation

In linear regression, we model the relationship between the dependent variable \( y \) and independent variables (features) \( X \) as:
\[
y = X\beta + \varepsilon
\]
where:
- \( y \in \mathbb{R}^n \) is the vector of target values.
- \( X \in \mathbb{R}^{n \times p} \) is the matrix of features. Often, the last column of \( X \) is set to 1 to incorporate the intercept.
- \( \beta \in \mathbb{R}^p \) is the vector of coefficients.
- \( \varepsilon \) is the error term.

The objective of OLS is to minimize the sum of squared errors:
\[
\min_{\beta} \; \| y - X\beta \|^2 = (y - X\beta)^\top (y - X\beta)
\]

### Derivation of the Normal Equations

To find the optimal \(\beta\), we set the derivative of the loss with respect to \(\beta\) to zero:
\[
\frac{\partial}{\partial \beta} \| y - X\beta \|^2 = -2X^\top (y - X\beta) = 0
\]
This yields the **normal equations**:
\[
X^\top X \beta = X^\top y
\]
Assuming \(X^\top X\) is invertible, the solution is:
\[
\beta = \left( X^\top X \right)^{-1} X^\top y
\]

#### Incorporating Regularization

To mitigate overfitting, we add a ridge regularization term (excluding the intercept) to the loss function:
\[
\min_{\beta} \; \| y - X\beta \|^2 + \lambda \|\beta_{\text{non-intercept}}\|^2
\]
The corresponding normal equations become:
\[
\left( X^\top X + \lambda I^* \right) \beta = X^\top y
\]
where \(I^*\) is an identity matrix modified to have a zero in the position corresponding to the intercept.

---

## Gradient Descent Optimization

When \(X^\top X\) is large or nearly singular, computing its inverse becomes impractical. Instead, **gradient descent** iteratively minimizes the loss.

### Standard Gradient Descent

For the unregularized loss function defined as:
\[
J(\beta) = \frac{1}{2n} \sum_{i=1}^{n} \left( y_i - x_i^\top \beta \right)^2,
\]
the gradient is:
\[
\nabla_\beta J(\beta) = -\frac{1}{n} X^\top (y - X\beta)
\]
The update rule for gradient descent is:
\[
\beta^{(t+1)} = \beta^{(t)} - \alpha \nabla_\beta J(\beta^{(t)})
\]
where \(\alpha\) is the learning rate.

### Ridge Regularization

When ridge regularization is included, the loss function becomes:
\[
J(\beta) = \frac{1}{2n} \sum_{i=1}^{n} \left( y_i - x_i^\top \beta \right)^2 + \frac{\lambda}{2} \sum_{j \neq \text{intercept}} \beta_j^2
\]
The gradient is then:
\[
\nabla_\beta J(\beta) = -\frac{1}{n} X^\top (y - X\beta) + \lambda \tilde{\beta}
\]
where \(\tilde{\beta}\) is the vector \(\beta\) with the intercept component zeroed out.

### Momentum in Gradient Descent

Momentum is used to accelerate gradient descent by smoothing updates. The update with momentum is given by:
1. Compute the gradient:
   \[
   g^{(t)} = \nabla_\beta J(\beta^{(t)})
   \]
2. Update the velocity:
   \[
   v^{(t+1)} = \gamma v^{(t)} + \alpha g^{(t)}
   \]
   where \(\gamma\) (typically between 0 and 0.9) is the momentum coefficient.
3. Update the parameters:
   \[
   \beta^{(t+1)} = \beta^{(t)} - v^{(t+1)}
   \]

The momentum term helps to overcome local minima and improves convergence speed by damping oscillations.

---

## Implementation Details

The repository is organized into several modules:

- **`main.py`**  
  Coordinates data generation, experiments, and plotting. It employs logging to track the experiment’s progress.

- **`lin_reg.py`**  
  Contains the `LinearRegression` class, which implements:
  - **OLS:** Solves the normal equations directly.
  - **Gradient Descent:** Iteratively updates \(\beta\) using gradient descent with support for early stopping, ridge regularization, and momentum.

- **`syn_data.py`**  
  Generates synthetic datasets using custom recurrence rules. The recurrence formulas simulate dependencies between features.

- **`plot_results.py`**  
  Provides functions to plot the OLS results (MSE vs. regularization parameter) and a heatmap of gradient descent performance over various hyperparameter combinations.

Each module uses detailed logging and type hints for improved maintainability and clarity.

---