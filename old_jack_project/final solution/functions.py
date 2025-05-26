#%% imports and functions definition
from sklearn import datasets
import pandas as pd
from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.datasets import make_regression, load_diabetes
from sklearn.impute import MissingIndicator
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import scipy.sparse
from sklearn.preprocessing import OneHotEncoder
from scipy.optimize import minimize_scalar
from typing import Tuple
# function that computes f(x)
def fLASSO(x, A, y):
    """
    Function that computes f(x) defined as: (1/2) * ||Ax - y||^2
    :param x: numpy array, the variable to optimize
    :param A: numpy array, the matrix in the constraint
    :param y: numpy array, the observed vector
    :return: f(x), scalar, the squared Euclidean norm (1/2) * ||Ax - y||^2
    """
    return 0.5 * np.linalg.norm(A @ x - y)**2

# function to compute gradient of f(x)
def gradient(x, A, y):
    """
    Function that computes the gradient of f(x) defined as: (1/2) * ||Ax - y||^2
    :param x: numpy array, the variable to optimize
    :param A: numpy array, the matrix in the constraint
    :param y: numpy array, the observed vector
    :return: gradient of f(x), numpy array of the same shape as x
    """
    if x.ndim == 1:
        x = x[:, np.newaxis]
    return A.T @ (A @ x - y)

def soft_thresholding(z, alpha):
    """
    Soft-thresholding operator
    
    z: vettore numpy, dimensione (p,1) o (p,)
    alpha: soglia (scalare)
    
    ritorna vettore numpy di stessa dimensione di z,
    applicando la formula:
        soft_threshold(z_i, alpha) = sign(z_i) * max(|z_i| - alpha, 0)
    """
    return np.sign(z) * np.maximum(np.abs(z) - alpha, 0)

def gradientDescentLASSO(A, y, l=500, tol=1e-4, K=15000):
    """
    Solves LASSO using (proximal) gradient descent
    -----------------------------------------------
    A: matrix (n x p)
    y: vector (n x 1) or (n,)
    l: lambda (L1 regularization penalty)
    tol: tolerance for stopping criterion
    K: maximum number of iterations

    Returns:
      data: list containing the objective function values (fLASSO + L1)
      k: number of actual iterations performed
      x: list of iterated solutions x[k]
      g: list of gradients (related to f(x))
    """
    n, p = A.shape
    L = np.linalg.norm(A, 2)**2
    # Gradient step size
    eta = 1.0 / L
    # Initialize variables
    x = [None] * (K + 1)
    g = [None] * (K + 1)
    data = [None] * (K + 1)
    # Initialize x[0] = 0
    x[0] = np.zeros((p, 1))
    # Initial evaluation of the objective function
    data[0] = fLASSO(x[0], A, y) + l * np.linalg.norm(x[0], 1)
    
    for k in range(1, K + 1):
        # Compute the gradient of the quadratic term
        g[k] = gradient(x[k-1], A, y)  # shape: (p,1)
        # Perform a gradient step
        #    x_tilde = x[k-1] - eta * g[k]
        x_tilde = x[k-1] - eta * g[k]
        # Apply soft-thresholding for L1 regularization
        #    x[k] = S_{eta*l}(x_tilde)
        x[k] = soft_thresholding(x_tilde, eta * l)
        # Compute the objective function
        data[k] = fLASSO(x[k], A, y) + l * np.linalg.norm(x[k], 1)
        # Stopping criterion (checking the objective function difference)
        if np.abs(data[k] - data[k-1]) < tol:
            print(f"Converged at iteration {k}")
            break
        print(f"Iteration {k}: improvement of {np.abs(data[k] - data[k-1])} with tolerance {tol}")
    
    return data, k, x, g


# function to find a point in the Oracle
def fwOracle(gradient, l):
    """
    Function that computes the Frank-Wolfe Oracle defined as:
        argmin(s) <gradient(f(x)), s> where s in the feasible
        set D and < > is the inner product.
    :param gradient: (p, 1) numpy vector
                Should be the gradient of f(x)
    :param l: (lambda) a scalar > 0
                Penalty parameter of the LASSO problem
    :return: s: (p, 1) numpy vector
                FW Oracle as defined above
    """
    # Initialize the zero vector
    p = gradient.shape[0]
    s = np.zeros((p, 1))
    # Check if the gradient is None or empty
    if gradient is None or gradient.size == 0:
        raise ValueError("Gradient cannot be None or empty in fwOracle.")
    # Check if all coordinates of x are 0
    # If they are, then the Oracle contains zero vector
    if (gradient != 0).sum() == 0:
        return s
    # Otherwise, follow the following steps
    # Trova la componente con valore assoluto maggiore
    a = abs(gradient)
    i = np.nonzero(a == max(a))[0][0]
    s[i] = - np.sign(gradient[i]) * l
    return s
    
# function for applying the Frank-Wolfe algorithm to solve LASSO problem
def frankWolfeLASSO(A, y, l=500, tol=1e-4, K=15000):
    """
    Applies the standard Frank-Wolfe algorithm to the LASSO problem (constrained formulation).
    """
    # Initialization
    n, p = A.shape
    x = [None] * K
    s = [None] * K
    g = [None] * K
    rho = [None] * K
    data = [None] * K
    duality_gaps = [None] * K
    x[0] = np.zeros((p, 1))
    s[0] = np.zeros((p, 1))
    data[0] = fLASSO(x[0], A, y)

    for k in range(1, K):
        # Compute gradient
        g[k] = gradient(x[k - 1], A, y)
        # Compute s_k via fwOracle
        s[k] = fwOracle(g[k], l)
        # Direzione d_k = s_k - x_{k-1}
        d_k = s[k] - x[k-1]
        # Line search per alpha in [0, 1]
        #    r_k = A x_{k-1} - y
        #    v_k = A (d_k)
        r_k = A @ x[k-1] - y
        v_k = A @ d_k
        # alpha* = - (r_k^T v_k)/(v_k^T v_k)
        num = np.dot(r_k.T, v_k)
        den = np.dot(v_k.T, v_k)
        if den == 0:
            # Se v_k = 0, la direzione non cambia nulla.
            alpha_star = 0
        else:
            alpha_star = - float(num / den)
        # Clamping to [0,1]
        rho[k] = max(0, min(1, alpha_star))
        # Update coefficients
        x[k] = (1 - rho[k]) * x[k - 1] + rho[k] * s[k]
        # Compute duality gap (exclude intercept)
        duality_gap = np.dot(g[k].T, x[k - 1] - s[k])
        duality_gaps[k] = duality_gap.item()
        # Record objective function value
        data[k] = fLASSO(x[k], A, y)
        # Check convergence
        if k > 1 and tol >= abs(duality_gaps[k]):
            print(f'Converged at iteration {k}')
            break
        print(f"iteration {k}: duality gap of {abs(duality_gaps[k])} and tolerance of {tol}")

    return data, k, x, s, g, duality_gaps

# Computes the Pairwise Frank-Wolfe Oracle
def pairwise_fwOracle(gradient, x, l):
    """
    Computes the Pairwise Frank-Wolfe Oracle for LASSO (constrained formulation).
    """

    p = gradient.shape[0]
    s = np.zeros((p, 1))  # Include intercept
    v = np.zeros((p, 1))  # Include intercept
    # Check if the gradient is None or empty
    if gradient is None or gradient.size == 0:
        raise ValueError("Gradient cannot be None or empty in fwOracle.")
    # Check if all coordinates of x are 0
    # If they are, then the Oracle contains zero vector
    if (gradient != 0).sum() == 0:
        return s
    # Otherwise, follow the following steps
    else:
        # Compute the (element-wise) absolute value of x
        a = np.abs(gradient)
        i = np.argmax(a)
        s[i] = -np.sign(gradient[i]) * l
        # Compute the Away Point
        norm_l1 = np.sum(np.abs(x))
        eps = 1e-12
        if abs(norm_l1 - l) < eps:
            # Esegui l'away step
            # cerchiamo la coordinata "attiva" j che massimizza grad_j * sign(x_j)
            # => j = argmax( gradient_j * sign(x_j) ), soggetto a x_j != 0
            active_idx = np.where(np.abs(x) > 0)[0]  # coordinate attive
            if len(active_idx) > 0:
                # calcoliamo i valori: away_val_j = gradient_j * sign(x_j)
                away_vals = gradient[active_idx] * np.sign(x[active_idx])
                j_away = active_idx[np.argmax(away_vals)]
                # v_j = sign(x_j) * l
                v[j_away] = np.sign(x[j_away]) * l
            else:
                # Se non ci sono coordinate attive (x=0 ma norm_l1 ~ l => contraddizione?), 
                # allora v=0
                pass
        else:
            # Non siamo sul boundary => no away step
            pass

    return s, v

# Applies the Pairwise Frank-Wolfe algorithm
def pairwiseFrankWolfeLASSO(A, y, l=500, tol=1e-4, K=15000):
    """
    Applies the Pairwise Frank-Wolfe algorithm to the LASSO problem (constrained formulation).
    """
    # Initialize
    n, p = A.shape
    x = [None] * K
    s = [None] * K
    v = [None] * K
    d = [None] * K
    g = [None] * K
    rho = [None] * K
    data = [None] * K
    duality_gaps = [None] * K
    x[0] = np.zeros((p, 1))
    s[0] = np.zeros((p, 1))
    v[0] = np.zeros((p, 1))
    data[0] = fLASSO(x[0], A, y)

    for k in range(1, K):
        # Compute gradient
        g[k] = gradient(x[k - 1], A, y)
        # Compute s_k and v_k via pairwise_fwOracle
        s[k], v[k] = pairwise_fwOracle(g[k], x[k-1], l)
        # Compute search direction d_k = s_k - v_k
        d[k] = s[k] - v[k]
        # 4) duality gap
        #    gap = - g[k]^T (s - x) per vanilla FW;
        #    pairwise => di solito gap = grad^T(x - s) o simile.
        #    Qui seguiamo la definizione "coerente" col vanilla: 
        #    grad^T (x - s) = - grad^T(s - x).
        #    Ma s - v non è la stessa di s - x. 
        #    Per coerenza, useremo => - g[k]^T d[k].
        duality_gaps[k] = - float(np.dot(g[k].T, d[k]))
        # Check convergence
        if abs(duality_gaps[k]) <= tol:
            print(f'Converged at iteration {k}')
            # Tronca le liste
            return data[:k+1], k, x[:k+1], s[:k+1], v[:k+1], d[:k+1], g[:k+1], duality_gaps[:k+1]
        # 6) line search "quadratica"
        r_k = A @ x[k-1] - y
        v_k = A @ d[k]
        num = - (r_k.T @ v_k)  # -g(k)^T d(k)
        den = (v_k.T @ v_k)
        if den <= 1e-15:
            alpha_quad = 1.0
        else:
            alpha_quad = float(num / den)
        # clamp in [0, 1]
        alpha_quad = max(0, min(1, alpha_quad))
        # 7) se x[k-1] + alpha_quad*d[k] esce dalla ball, 
        #    riduciamo alpha con bisezione
        def norm_l1_feasible(alpha):
            x_temp = x[k-1] + alpha * d[k]
            return np.sum(np.abs(x_temp)) <= l
        if norm_l1_feasible(alpha_quad):
            alpha = alpha_quad
        else:
            # bisection su [0, alpha_quad]
            left, right = 0.0, alpha_quad
            for _ in range(30):  # 30 iter max bisezione
                mid = 0.5*(left+right)
                if norm_l1_feasible(mid):
                    left = mid
                else:
                    right = mid
            alpha = 0.5*(left+right)
        rho[k] = alpha
        # 8) Update x
        x[k] = x[k-1] + alpha * d[k]
        # 9) obiettivo
        data[k] = fLASSO(x[k], A, y)
        print(f"iteration {k}: duality gap of {abs(duality_gaps[k])} and tolerance of {tol}")

    return data, K-1, x, s, v, d, g, duality_gaps

def gridSearch_FW(lambdas, variant, X_train, X_test, y_train, y_test):
    # Lists to store results
    results_grid_search = []
    all_coefficients = []  # Store coefficients for each lambda
    mse_values = []  # Store MSE values for each lambda

    # Iterate over each lambda value
    for l in lambdas:
        print(f"Running Frank-Wolfe for lambda={l}")
        
        # Execute the Frank-Wolfe algorithm for Lasso with the current lambda value
        if variant == "vanilla":
            data, k, x, s, g, duality_gaps = frankWolfeLASSO(X_train, y_train, l=l, tol=1000, K=10000)
        elif variant == "pairwise":
            data, k, x, s, v, d, g, duality_gaps = pairwiseFrankWolfeLASSO(X_train, y_train, l=l, tol=1000, K=10000)
        elif variant == "standard_lasso":
            data, k, x, g = gradientDescentLASSO(X_train, y_train, l=l, tol=10000, K=10000)
        
        # Retrieve the coefficients from the Frank-Wolfe result
        coefficients = x[k - 1].flatten()  # Extract coefficients from the last iteration
        
        # Compute y_pred using the coefficients and X_test
        y_pred = np.dot(X_test, coefficients)
        y_pred = y_pred.reshape(-1, 1)
        
        # Compute evaluation metrics using a custom function
        results = evaluate_model_performance(X_test, y_pred, y_test, n_iterations=k)
        
        # Store grid search results
        results_grid_search.append({
            'lambda': l,
            'coefficients': coefficients,
            'n_iterations': results['n_iterations'],
            'total_operations': results['total_operations'],
            'mse': results['mse'],
            'mae': results['mae'],
            'r2': results['r2'],
            'adjusted_r2': results['adjusted_r2']
        })
        
        # Store coefficients and MSE values
        all_coefficients.append(coefficients)
        mse_values.append(results['mse'])
        
        # Display results for this lambda value
        print(f"Results for lambda={l}:")
        print("Number of iterations:", results['n_iterations'])
        print("Total operations:", results['total_operations'])
        print("MSE:", results['mse'])
        print("MAE:", results['mae'])
        print("R²:", results['r2'])
        print("Adjusted R²:", results['adjusted_r2'])
        print("-" * 50)
    
    # After grid search, select the best model (e.g., the one with the lowest MSE)
    if variant == "standard_lasso":
        best_result = min(
            reversed(results_grid_search),
            key=lambda x: round(x['mse'], 2)
        )
    else:
        best_result = min(results_grid_search,
            key=lambda x: round(x['mse'], 2)
        )
    
    best_lambda = best_result['lambda']

    print("\n\n\nBest result from grid search:")
    print(f"Lambda: {best_lambda}")
    print("MSE:", best_result['mse'])
    print("MAE:", best_result['mae'])
    print("R²:", best_result['r2'])
    print("Adjusted R²:", best_result['adjusted_r2'])
    
    # Convert all_coefficients to an array for easier manipulation
    all_coefficients = np.array(all_coefficients)
    
    # Plot the trend of coefficients with respect to lambda
    plt.figure(figsize=(8, 8))
    for i in range(all_coefficients.shape[1]):
        plt.plot(lambdas, all_coefficients[:, i], label=f'Coeff {i+1}')
    plt.axvline(x=best_lambda, color='red', linestyle='--', label='Best Lambda')
    plt.xscale('log')
    plt.xlabel('Lambda')
    plt.ylabel('Coefficient Value')
    plt.title('Coefficients with respect to Lambda')
    plt.show()

    # Plot the trend of deviance (MSE) with respect to lambda
    plt.figure(figsize=(8, 8))
    plt.plot(lambdas, mse_values, marker='o', linestyle='-', label='MSE')
    plt.axvline(x=best_lambda, color='red', linestyle='--', label='Best Lambda')
    plt.xscale('log')
    plt.xlabel('Lambda')
    plt.ylabel('MSE')
    plt.title('Deviance (MSE) with respect to Lambda')
    plt.legend()
    plt.show()

    return best_lambda


# REMEMBER TO COMMENT THE REST AND TRANSLATE MISSING ITALIAN BITS TO ENG!!
def check_data_format(A, y):
    """
    Verifies if X and y are in the correct format and prints key characteristics.
    
    Parameters:
    A: numpy array
        Feature matrix of shape (n, p), where n is the number of observations and p is the number of parameters.
    y: numpy array
        Response vector of shape (n, 1), where n is the number of observations.
    
    Returns:
    None
    """
    n_A, p = None, None
    
    if isinstance(A, np.ndarray):
        if A.ndim == 2:
            n_A, p = A.shape
            print(f"The feature matrix X is correctly formatted.")
            print(f"Shape of X: {A.shape}")
            print(f"Number of observations (n): {n_A}")
            print(f"Number of parameters (p): {p}")
        else:
            print(f"Error: X should be two-dimensional, but has {A.ndim} dimensions.")
            print(f"Data type of X: {type(A)}")
            print(f"Shape of X: {A.shape}")
    else:
        print(f"Error: X is not a NumPy array. It is of type: {type(A)}")

    # Verify that y is a two-dimensional column vector
    if isinstance(y, np.ndarray) and y.ndim == 2 and y.shape[1] == 1:
        n_y = y.shape[0]  # Get the number of observations in y
        if n_A is not None and n_y == n_A:
            print(f"The response vector y is correctly formatted.")
            print(f"Shape of y: {y.shape}")
            print(f"Number of observations (n): {n_y}")
        else:
            print(f"Error: The number of observations in y ({n_y}) does not match the number of observations in A ({n_A}).")
    else:
        print("Error: y must be a two-dimensional column vector with shape (n, 1).")
        print(f"Data type of y: {type(y)}")
        print(f"Shape of y: {y.shape}")


def check_scaling(X, tol_mean=1e-2, tol_std=1e-2):
    """
    Checks if the matrix X is properly scaled.
    
    Parameters:
    X (numpy array): The feature matrix to be checked.
    tol_mean (float): Tolerance for deviation of the mean from 0.
    tol_std (float): Tolerance for deviation of the standard deviation from 1.
    
    Returns:
    bool: True if X is properly scaled, False otherwise.
    """
    # Compute the mean and standard deviation for each column
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    
    # Check if the mean is close to 0
    mean_check = np.all(np.abs(means) < tol_mean)
    
    # Check if the standard deviation is close to 1
    std_check = np.all(np.abs(stds - 1) < tol_std)
    
    if mean_check and std_check:
        print("The feature matrix X is properly scaled.")
        return True
    else:
        print("The feature matrix X is NOT properly scaled.")
        if not mean_check:
            print("The following columns have a mean significantly different from 0:")
            print(np.where(np.abs(means) >= tol_mean))
        if not std_check:
            print("The following columns have a standard deviation significantly different from 1:")
            print(np.where(np.abs(stds - 1) >= tol_std))
        return False

    
def evaluate_model_performance(X_test, y_pred, y_test, n_iterations, tol=1e-4):
    """
    Computes various model performance metrics and iteration details.

    :param X_test: numpy array, test data (features)
    :param y_pred: numpy array, predicted values
    :param y_test: numpy array, actual target values
    :param n_iterations: int, number of iterations until convergence
    :param tol: float, tolerance for determining convergence
    :return: dict, computed metrics
    """
    
    # Ensure that both prediction and target vectors are flattened
    y_pred = y_pred.flatten()
    y_test = y_test.flatten()
    
    # Compute MSE, MAE, and R²
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Compute adjusted R²
    n = X_test.shape[0]  # Number of observations
    p = X_test.shape[1]  # Number of predictors (including intercept, if present)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    # Count operations (multiplications and additions) per iteration
    n_operations_per_iteration = X_test.shape[0] * X_test.shape[1] * 2  # Multiplications and additions
    total_operations = n_operations_per_iteration * n_iterations
    
    # Results
    results = {
        'n_iterations': n_iterations,
        'total_operations': total_operations,
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'adjusted_r2': adj_r2
    }
    
    return results



def ames_preprocessing(X_train, X_test, y_train, y_test, relevant_columns):
    
    X_train["FireplaceQu"] = X_train["FireplaceQu"].fillna("N/A")
    X_train["FireplaceQu"].value_counts()
    # (1) Identify data to be transformed
    # We only want missing indicators for LotFrontage
    frontage_train = X_train[["LotFrontage"]]
    # (2) Instantiate the transformer object
    missing_indicator = MissingIndicator()
    # (3) Fit the transformer object on frontage_train
    missing_indicator.fit(frontage_train)
    # (4) Transform frontage_train and assign the result
    # to frontage_missing_train
    frontage_missing_train = missing_indicator.transform(frontage_train)
    # Visually inspect frontage_missing_train
    #frontage_missing_train

    # frontage_missing_train should be a NumPy array
    assert type(frontage_missing_train) == np.ndarray
    # We should have the same number of rows as the full X_train
    assert frontage_missing_train.shape[0] == X_train.shape[0]
    # But we should only have 1 column
    assert frontage_missing_train.shape[1] == 1
    X_train["LotFrontage_Missing"] = frontage_missing_train
    # (1) frontage_train was created previously, so we don't
    # need to extract the relevant data again
    # (2) Instantiate a SimpleImputer with strategy="median"
    imputer = SimpleImputer(strategy="median")
    # (3) Fit the imputer on frontage_train
    imputer.fit(frontage_train)
    # (4) Transform frontage_train using the imputer and
    # assign the result to frontage_imputed_train
    frontage_imputed_train = imputer.transform(frontage_train)
    # (5) Replace value of LotFrontage
    X_train["LotFrontage"] = frontage_imputed_train
    print(X_train.shape)
    #assert X_train.shape == (1095, 16)
    X_train.isna().sum()
    print(X_train["Street"].value_counts())
    print()
    print(X_train["FireplaceQu"].value_counts())
    print()
    print(X_train["LotFrontage_Missing"].value_counts())
    # (1) Create a variable street_train that contains the relevant column from X_train
    # (Use double brackets [[]] to get the appropriate shape)
    street_train = X_train[["Street"]]
    # (2) Instantiate an OrdinalEncoder
    encoder_street = OrdinalEncoder()
    # (3) Fit the encoder on street_train
    encoder_street.fit(street_train)
    # (4) Transform street_train using the encoder and
    # assign the result to street_encoded_train
    street_encoded_train = encoder_street.transform(street_train)
    # Flatten for appropriate shape
    street_encoded_train = street_encoded_train.flatten()
    # Visually inspect street_encoded_train
    street_encoded_train
    # (5) Replace value of Street
    X_train["Street"] = street_encoded_train
    # (1) We already have a variable frontage_missing_train
    # from earlier, no additional step needed
    # (2) Instantiate an OrdinalEncoder for missing frontage
    encoder_frontage_missing = OrdinalEncoder()
    # (3) Fit the encoder on frontage_missing_train
    encoder_frontage_missing.fit(frontage_missing_train)
    # Inspect the categories of the fitted encoder
    encoder_frontage_missing.categories_[0]
    # (4) Transform frontage_missing_train using the encoder and
    # assign the result to frontage_missing_encoded_train
    frontage_missing_encoded_train = encoder_frontage_missing.transform(frontage_missing_train)
    # Flatten for appropriate shape
    frontage_missing_encoded_train = frontage_missing_encoded_train.flatten()
    # (5) Replace value of LotFrontage_Missing
    X_train["LotFrontage_Missing"] = frontage_missing_encoded_train
    # (0) import OneHotEncoder from sklearn.preprocessing
    # (1) Create a variable fireplace_qu_train
    # extracted from X_train
    # (double brackets due to shape expected by OHE)
    fireplace_qu_train = X_train[["FireplaceQu"]]
    # (2) Instantiate a OneHotEncoder with categories="auto",
    # sparse=False, and handle_unknown="ignore"
    ohe = OneHotEncoder(categories="auto", sparse_output=False, handle_unknown="ignore")
    # (3) Fit the encoder on fireplace_qu_train
    ohe.fit(fireplace_qu_train)
    # (4) Transform fireplace_qu_train using the encoder and
    # assign the result to fireplace_qu_encoded_train
    fireplace_qu_encoded_train = ohe.transform(fireplace_qu_train)
    # (5a) Make the transformed data into a dataframe
    fireplace_qu_encoded_train = pd.DataFrame(
        # Pass in NumPy array
        fireplace_qu_encoded_train,
        # Set the column names to the categories found by OHE
        columns=ohe.categories_[0],
        # Set the index to match X_train's index
        index=X_train.index
    )
    # (5b) Drop original FireplaceQu column
    X_train.drop("FireplaceQu", axis=1, inplace=True)
    # (5c) Concatenate the new dataframe with current X_train
    X_train = pd.concat([X_train, fireplace_qu_encoded_train], axis=1)
    X_test = X_test.loc[:, relevant_columns]
    # Replace FireplaceQu NaNs with "N/A"s
    X_test["FireplaceQu"] = X_test["FireplaceQu"].fillna("N/A")
    # Add missing indicator for lot frontage
    frontage_test = X_test[["LotFrontage"]]
    frontage_missing_test = missing_indicator.transform(frontage_test)
    X_test["LotFrontage_Missing"] = frontage_missing_test
    # Impute missing lot frontage values
    frontage_imputed_test = imputer.transform(frontage_test)
    X_test["LotFrontage"] = frontage_imputed_test
    # Check that there are no more missing values
    X_test.isna().sum()
    # Encode street type
    street_test = X_test[["Street"]]
    street_encoded_test = encoder_street.transform(street_test).flatten()
    X_test["Street"] = street_encoded_test
    # Encode frontage missing
    frontage_missing_test = X_test[["LotFrontage_Missing"]]
    frontage_missing_encoded_test = encoder_frontage_missing.transform(frontage_missing_test).flatten()
    X_test["LotFrontage_Missing"] = frontage_missing_encoded_test
    # One-hot encode fireplace quality
    fireplace_qu_test = X_test[["FireplaceQu"]]
    fireplace_qu_encoded_test = ohe.transform(fireplace_qu_test)
    fireplace_qu_encoded_test = pd.DataFrame(
        fireplace_qu_encoded_test,
        columns=ohe.categories_[0],
        index=X_test.index
    )
    X_test.drop("FireplaceQu", axis=1, inplace=True)
    X_test = pd.concat([X_test, fireplace_qu_encoded_test], axis=1)

    X_train = X_train.to_numpy()  # O usa .values se stai usando una versione di Pandas più vecchia
    X_test = X_test.to_numpy()
    #X_train = X_train.values
    y_train = y_train.reshape(-1, 1)
    #X_test = X_test.values
    y_test = y_test.reshape(-1, 1)

    return X_train, X_test, y_train, y_test

def dataset_loading_and_preprocessing(which_dataset:str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if which_dataset == "LARS":
        lars_X, lars_y = datasets.make_regression(n_samples=4000, n_features=200, n_informative=20, noise=0.5, random_state=42)
        lars_df = pd.DataFrame(lars_X)
        lars_df['TARGET'] = lars_y  # Aggiungiamo la colonna target
        X = lars_df.drop(columns=['TARGET']).to_numpy()  # Caratteristiche
        y = lars_df['TARGET'].to_numpy().reshape(-1, 1)  # Target

    elif which_dataset == "WINE":
        wine_data = fetch_openml(name='wine-quality-white', version=1, as_frame=True)
        X = pd.DataFrame(wine_data.data, columns=wine_data.feature_names).to_numpy()
        y = pd.to_numeric(wine_data.target, errors='coerce').to_numpy().reshape(-1, 1) # Converte y in numerico e poi in un array 2D

    elif which_dataset == "AMES":
        ames = fetch_openml(name="house_prices", as_frame=True)
        ames_df = pd.DataFrame(ames.data, columns=ames.feature_names)
        ames_df['SalePrice'] = ames.target  # Aggiungiamo la colonna target
        X = ames_df.drop(columns=['SalePrice'])  # Caratteristiche
        y = ames_df['SalePrice'].astype(int).values  # Target
        # Declare relevant columns
        relevant_columns = [
            'LotFrontage',  # Linear feet of street connected to property
            'LotArea',      # Lot size in square feet
            'Street',       # Type of road access to property
            'OverallQual',  # Rates the overall material and finish of the house
            'OverallCond',  # Rates the overall condition of the house
            'YearBuilt',    # Original construction date
            'YearRemodAdd', # Remodel date (same as construction date if no remodeling or additions)
            'GrLivArea',    # Above grade (ground) living area square feet
            'FullBath',     # Full bathrooms above grade
            'BedroomAbvGr', # Bedrooms above grade (does NOT include basement bedrooms)
            'TotRmsAbvGrd', # Total rooms above grade (does not include bathrooms)
            'Fireplaces',   # Number of fireplaces
            'FireplaceQu',  # Fireplace quality
            'MoSold',       # Month Sold (MM)
            'YrSold'        # Year Sold (YYYY)
        ]
        # Reassign X_train so that it only contains relevant columns
        X = X[relevant_columns]

    elif which_dataset == "YPMSDW":
        import urllib.request
        import zipfile
        import os
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip"
        zip_path = "YearPredictionMSD.zip"
        data_path = "YearPredictionMSD.txt"
        if not os.path.exists(data_path):
            urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(".")
            os.remove(zip_path)
        data = pd.read_csv(data_path, header=None)
        X = data.iloc[:, 1:].to_numpy()  # Tutte le colonne eccetto la prima
        y = data.iloc[:, 0].to_numpy().reshape(-1, 1)  # La prima colonna come variabile target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    if which_dataset == "AMES":
            X_train, X_test, y_train, y_test = ames_preprocessing(X_train, X_test, y_train, y_test, relevant_columns)


    # Scalatura di X
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)

    """if which_dataset != "LARS":
        scaler_y = StandardScaler()
    else:
        scaler_y = MinMaxScaler()

    y_train = scaler_y.fit_transform(y_train)
    y_test = scaler_y.transform(y_test) """

    X_train = np.c_[np.ones((X_train.shape[0], 1)), X_train]
    X_test = np.c_[np.ones((X_test.shape[0], 1)), X_test]
    
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    
    return X_train, X_val, X_test, y_train, y_val, y_test 