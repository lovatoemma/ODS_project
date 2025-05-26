#%% IMPORTS
import pandas as pd
import numpy as np
from functions import check_scaling, check_data_format, frankWolfeLASSO, pairwiseFrankWolfeLASSO, gridSearch_FW, evaluate_model_performance, dataset_loading_and_preprocessing, gradientDescentLASSO
import matplotlib.pyplot as plt


#%% DATA LOADING AND PREPROCESSING:
available_datasets = ["LARS", "WINE", "AMES", "YPMSDW"]
which_dataset_toload = available_datasets[3]
X_train, X_val, X_test, y_train, y_val, y_test = dataset_loading_and_preprocessing(which_dataset=which_dataset_toload)
is_scaled = check_scaling(X_train)
check_data_format(X_train, y_train)
check_data_format(X_test, y_test)

#%% GRID SEARCH FOR BEST LAMBDA PARAMETER for STANDARD LASSO 
# Define lambda values (regularization parameter) for grid search
lambdas = np.logspace(0, 8, 10)  # From 0.01 to 100 on a logarithmic scale
best_lambda_gradientDescentLASSO = gridSearch_FW(lambdas=lambdas, variant="standard_lasso", 
                                                 X_train=X_train, X_test=X_val, 
                                                 y_train=y_train, y_test=y_val)

#%% FW TRAIN AND EVAL for best lambda
data, k, x, g = gradientDescentLASSO(X_train, y_train, l=best_lambda_gradientDescentLASSO, 
                                     tol=1e-2, K=10000)

# Retrieve the coefficients from the Frank-Wolfe optimization result
coefficients = x[k - 1].flatten()  # Extract the coefficients from the last iteration

# Compute predicted values (y_pred) using the obtained coefficients and X_test
y_pred = np.dot(X_test, coefficients)
y_pred = y_pred.reshape(-1, 1)

# Compute evaluation metrics using a custom function
results = evaluate_model_performance(X_test, y_pred, y_test, n_iterations=k)

# Display results
print(coefficients)
print("Number of iterations:", results['n_iterations'])
print("Total operations:", results['total_operations'])
print("MSE:", results['mse'])
print("MAE:", results['mae'])
print("R²:", results['r2'])
print("Adjusted R²:", results['adjusted_r2'])


#%% Print the dimensions of y_test and y_pred
print("Shape of y_test:", y_test.shape)
print("Shape of y_pred:", y_pred.shape)

# Recalculate residuals
residuals = y_test - y_pred

# Verify that the dimensions of residuals and predictions match
print("Shape of residuals:", residuals.shape)

# Check if the dimensions are equal before plotting
if y_pred.shape == residuals.shape:
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot')
    plt.show()
else:
    print("The dimensions of y_pred and residuals do not match.")

#%% GRID SEARCH FOR BEST LAMBDA PARAMETER for FW LASSO
# Define lambda values (regularization parameter) for grid search
lambdas = np.logspace(0, 6, 10)  # From 0.01 to 100 on a logarithmic scale
best_lambda_vanilla_FW = gridSearch_FW(lambdas=lambdas, variant="vanilla", 
                                       X_train=X_train, X_test=X_val, 
                                       y_train=y_train, y_test=y_val)

#%% FW TRAIN AND EVAL for best lambda
data, k, x, s, g, duality_gaps = frankWolfeLASSO(X_train, y_train, l=best_lambda_vanilla_FW, 
                                                 tol=100, K=10000)

# Retrieve the coefficients from the Frank-Wolfe optimization result
coefficients = x[k - 1].flatten()  # Extract the coefficients from the last iteration

# Compute predicted values (y_pred) using the obtained coefficients and X_test
y_pred = np.dot(X_test, coefficients)
y_pred = y_pred.reshape(-1, 1)

# Compute evaluation metrics using a custom function
results = evaluate_model_performance(X_test, y_pred, y_test, n_iterations=k)

# Display results
print(coefficients)
print("Number of iterations:", results['n_iterations'])
print("Total operations:", results['total_operations'])
print("MSE:", results['mse'])
print("MAE:", results['mae'])
print("R²:", results['r2'])
print("Adjusted R²:", results['adjusted_r2'])


#%% Print the dimensions of y_test and y_pred
print("Shape of y_test:", y_test.shape)
print("Shape of y_pred:", y_pred.shape)

# Recalculate residuals
residuals = y_test - y_pred

# Verify that the dimensions of residuals and predictions match
print("Shape of residuals:", residuals.shape)

# Check if the dimensions are equal before plotting
if y_pred.shape == residuals.shape:
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot')
    plt.show()
else:
    print("The dimensions of y_pred and residuals do not match.")

# Define the range of iterations for plotting duality gaps
iterations = range(1, len(duality_gaps))  # From 1 to the number of iterations



#%% PAIRWISE FW GRID SEARCH FOR BEST LAMBDA
# Define lambda values (regularization parameter) for grid search
lambdas = np.logspace(0, 6, 10)  # From 0.01 to 100 on a logarithmic scale
best_lambda_pairwise_FW = gridSearch_FW(lambdas=lambdas, variant="pairwise", 
                                        X_train=X_train, X_test=X_val, 
                                        y_train=y_train, y_test=y_val)

#%% BEST PAIRWISE FW TRAIN AND EVAL:
data, k, x, s, v, d, g, duality_gaps_p = pairwiseFrankWolfeLASSO(X_train, y_train, 
                                                                 l=best_lambda_pairwise_FW, 
                                                                 tol=100, K=10000)

# Retrieve the coefficients from the Pairwise Frank-Wolfe optimization result
coefficients = x[k - 1].flatten()  # Extract the coefficients from the last iteration

# Compute predicted values (y_pred) using the obtained coefficients and X_test
y_pred = np.dot(X_test, coefficients)
y_pred = y_pred.reshape(-1, 1)

# Compute evaluation metrics using a custom function
results = evaluate_model_performance(X_test, y_pred, y_test, n_iterations=k)

# Display results
print(coefficients)
print("Number of iterations:", results['n_iterations'])
print("Total operations:", results['total_operations'])
print("MSE:", results['mse'])
print("MAE:", results['mae'])
print("R²:", results['r2'])
print("Adjusted R²:", results['adjusted_r2'])


#%% Print the dimensions of y_test and y_pred
print("Shape of y_test:", y_test.shape)
print("Shape of y_pred:", y_pred.shape)

# Recalculate residuals
residuals = y_test - y_pred

# Verify that the dimensions of residuals and predictions match
print("Shape of residuals:", residuals.shape)

# Check if the dimensions are equal before plotting
if y_pred.shape == residuals.shape:
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot')
    plt.show()
else:
    print("The dimensions of y_pred and residuals do not match.")

#%%
# Define the range of iterations for plotting duality gaps
# iterations = range(1, len(duality_gaps))  # From 1 to the number of iterations

# Create the duality gap convergence plot
plt.figure(figsize=(12, 8))
plt.plot(iterations, duality_gaps[1:], marker='*', label='Standard Frank-Wolfe')
plt.plot(duality_gaps_p[1:], marker='o', label='Pairwise Frank-Wolfe')
plt.yscale("log")
plt.xlabel('Iterations')
plt.ylabel('Duality Gaps')
plt.title('Duality Gap Convergence During Iterations')
plt.grid(True)
plt.legend()  # Display the legend for both curves
plt.show()
# %%

