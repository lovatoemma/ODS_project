import numpy as np
from numpy.linalg import svd
import matplotlib.pyplot as plt


def initialize_matrix(n_rows, n_cols, rank, observed_matrix):
    """
    Initialize a low-rank matrix randomly and normalize it based on the observed matrix.
    
    Parameters:
        n_rows (int): Number of rows.
        n_cols (int): Number of columns.
        rank (int): Rank of the low-rank approximation.
        observed_matrix (numpy.ndarray): The original matrix with missing values as np.nan.

    Returns:
        numpy.ndarray: Normalized initialized matrix.
    """
    # Random low-rank initialization
    U = np.random.rand(n_rows, rank)
    V = np.random.rand(n_cols, rank)
    X0 = np.dot(U, V.T)

    # Compute statistics of observed values 
    observed_mean = np.nanmean(observed_matrix)
    observed_std = np.nanstd(observed_matrix)
    # Normalize X0 to match the observed mean and standard deviation
    X0_mean = np.mean(X0)
    X0_std = np.std(X0)

    if X0_std > 0:  
        X0 = (X0 - X0_mean) / X0_std * observed_std + observed_mean
    else:
        X0 = np.full_like(X0, observed_mean) 

    return X0

def compute_loss(observed_indices, observed_values, completed_matrix):
    """Compute the loss for observed entries."""
    predictions = completed_matrix[observed_indices]
    return 0.5 * np.sum((predictions - observed_values) ** 2)

def compute_gradient(observed_indices, observed_values, completed_matrix):
    """Compute the gradient with respect to observed entries."""
    grad = np.zeros_like(completed_matrix)
    for (i, j), value in zip(zip(*observed_indices), observed_values):
        grad[i, j] = completed_matrix[i, j] - value
    return 2*grad


def line_search(completed_matrix, low_rank_update, observed_indices, observed_values, num_steps=1000):
    """
    Performs a line search to find the optimal step size that minimizes the loss.

    Parameters:
        completed_matrix (np.ndarray): Current matrix estimate.
        low_rank_update (np.ndarray): Low-rank update direction.
        observed_indices (tuple of arrays): Indices of observed entries.
        observed_values (array): Values of observed entries.
        num_steps (int): Number of steps to try in the line search.

    Returns:
        float: Optimal step size.
    """
    step_size = 0
    min_loss = float('inf')
    for step in np.linspace(0, 1, num_steps):
        candidate = (1 - step) * completed_matrix + step * low_rank_update
        loss = compute_loss(observed_indices, observed_values, candidate)
        if loss < min_loss:
            min_loss = loss
            step_size = step
    return step_size

def armijo_rule(completed_matrix, low_rank_update, observed_indices, observed_values, grad, beta=0.5, gamma=0.1):
    """
    Determines the step size using the Armijo rule for sufficient decrease.

    Parameters:
        completed_matrix (np.ndarray): Current matrix estimate.
        low_rank_update (np.ndarray): Low-rank update direction.
        observed_indices (tuple of arrays): Indices of observed entries.
        observed_values (array): Values of observed entries.
        grad (np.ndarray): Gradient of the loss function.
        beta (float): Step size reduction factor.
        gamma (float): Armijo rule parameter.

    Returns:
        float: Step size satisfying the Armijo condition.
    """
    step_size = 1.0  
    while True:
        candidate = (1 - step_size) * completed_matrix + step_size * low_rank_update
        lhs = compute_loss(observed_indices, observed_values, candidate)
        rhs = compute_loss(observed_indices, observed_values, completed_matrix) + gamma * step_size * np.trace(np.matmul(grad.T, low_rank_update - completed_matrix))
        if lhs <= rhs:
            break
        step_size *= beta  
    return step_size

def frank_wolfe_matrix_completion(n_rows, n_cols, observed_indices, observed_values,
                                   rank=1, max_iter=100, sigma=1, stopping_conditions=None, 
                                   step_size_method='line_search', print_freq=1):
    """
    Frank-Wolfe algorithm for matrix completion with multiple stopping conditions.

    Parameters:
        n_rows (int): Number of rows in the target matrix.
        n_cols (int): Number of columns in the target matrix.
        observed_indices (tuple of arrays): Indices of observed entries.
        observed_values (array): Values of observed entries.
        rank (int): Target rank for the solution.
        max_iter (int): Maximum number of iterations.
        sigma (float): Scaling factor for the low-rank update.
        stopping_conditions (dict or None): Stopping criteria:
            - 'loss': Stop if loss function is below threshold.
            - 'dual_gap': Stop if dual gap is below threshold.
            - 'rel_change': Stop if relative change in matrix is below threshold.
        step_size_method (str): Method for determining step size ('diminishing', 'line_search', 'armijo').
        print_freq (int): Frequency of printing progress updates (e.g., every `print_freq` iterations).

    Returns:
        np.ndarray: Completed matrix.
        list: List of loss values at each iteration.
        list: List of dual gap values at each iteration.
    """
    completed_matrix = initialize_matrix(n_rows, n_cols, rank, observed_values)
    prev_matrix = np.copy(completed_matrix)
    losses = []
    dual_gaps = []
    
    for iteration in range(max_iter):
        grad = compute_gradient(observed_indices, observed_values, completed_matrix)
        U, S, Vt = svd(-grad, full_matrices=False)
        low_rank_update = sigma * np.outer(U[:, 0], Vt[0, :])
        
        if step_size_method == 'diminishing':
            step_size = 2 / (iteration + 2)
        elif step_size_method == 'line_search':
            step_size = line_search(completed_matrix, low_rank_update, observed_indices, observed_values)
        elif step_size_method == 'armijo':
            step_size = armijo_rule(completed_matrix, low_rank_update, observed_indices, observed_values, grad)
        else:
            raise ValueError("Invalid step_size_method. Choose from 'diminishing', 'line_search', or 'armijo'")
        
        completed_matrix = (1 - step_size) * completed_matrix + step_size * low_rank_update
        
        loss = compute_loss(observed_indices, observed_values, completed_matrix)
        dual_gap = np.trace(np.matmul(grad.T, prev_matrix - low_rank_update))
        rel_change = np.linalg.norm(completed_matrix - prev_matrix) / np.linalg.norm(prev_matrix)
        
        losses.append(loss)
        dual_gaps.append(dual_gap)
        
        if iteration % print_freq == 0:
            print(f"Iteration {iteration + 1}, Loss: {loss}, Dual Gap: {dual_gap}, Rel Change: {rel_change}, Step Size: {step_size}")
        
        if stopping_conditions:
            if 'loss' in stopping_conditions and loss < stopping_conditions['loss']:
                print("Stopping: Loss threshold reached.")
                break
            if 'rel_change' in stopping_conditions and rel_change < stopping_conditions['rel_change']:
                print("Stopping: Relative change threshold reached.")
                break
        
        prev_matrix = np.copy(completed_matrix)
    
    return completed_matrix, losses, dual_gaps


def plot_losses_and_gaps(losses, gaps):
    """
    Plots the loss function and dual gap over iterations in a vertically stacked format.
    
    Parameters:
        losses (list): List of loss function values over iterations.
        gaps (list): List of dual gap values over iterations.
    """
    iterations = range(1, len(losses) + 1)
    
    fig, ax = plt.subplots(2, 1, figsize=(6, 10), sharex=True)
    
    # Plot loss function
    ax[0].plot(iterations, losses, linestyle='-', label='Loss')
    ax[0].set_ylabel("Loss Function")
    ax[0].set_title("Loss Function Over Iterations")
    ax[0].legend()
    ax[0].grid(True)
    
    # Plot dual gap
    ax[1].plot(iterations, gaps, linestyle='-', color='r', label='Dual Gap')
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel("Dual Gap")
    ax[1].set_title("Dual Gap Over Iterations")
    ax[1].legend()
    ax[1].grid(True)
    
    plt.tight_layout()
    plt.show()


def plot_losses_and_gaps_comparison(step_size_methods, losses_dict, gaps_dict):
    """
    Plots the loss function and dual gap over iterations for different step size methods.
    
    Parameters:
        step_size_methods (list): List of step size method names.
        losses_dict (dict): Dictionary where keys are step size method names and values are lists of loss function values.
        gaps_dict (dict): Dictionary where keys are step size method names and values are lists of dual gap values.
    """
    fig, ax = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
    
    # Plot loss function for each step size method
    for method in step_size_methods:
        iterations = range(1, len(losses_dict[method]) + 1)
        ax[0].plot(iterations, losses_dict[method], linestyle='-', label=method)
    ax[0].set_ylabel("Loss Function")
    ax[0].set_title("Loss Function Over Iterations")
    ax[0].legend()
    ax[0].grid(True)
    
    # Plot dual gap for each step size method
    for method in step_size_methods:
        iterations = range(1, len(gaps_dict[method]) + 1)
        ax[1].plot(iterations, gaps_dict[method], linestyle='-', label=method)
    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel("Dual Gap")
    ax[1].set_title("Dual Gap Over Iterations")
    ax[1].legend()
    ax[1].grid(True)
    
    plt.tight_layout()
    plt.show()