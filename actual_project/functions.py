import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds

# Problem Definition 
class MatrixCompletionProblem:
    """
    Logic logic for the Matrix Completion problem.
    f(X) = ||P_J(X) - P_J(U)||_F^2
    """
    def __init__(self, U_observed: csc_matrix):
        """
        Initializes the problem.
        Args:
            U_observed: A sparse matrix containing only the observed entries of U.
        """
        self.U_observed = U_observed
        self.m, self.n = U_observed.shape
        # P_J(U) in dense format for easier calculations
        self.U_dense_observed = U_observed.toarray()
        # Mask for the projection P_J
        self.omega_mask = (self.U_dense_observed != 0)

    def project_on_omega(self, A: np.ndarray) -> np.ndarray:
        """
        Projects a matrix A onto the set of observed indices J (Omega).
        """
        return A * self.omega_mask

    def objective_function(self, X: np.ndarray) -> float:
        """
        Calculates the value of the objective function f(X).
        """
        residual = self.project_on_omega(X - self.U_dense_observed)
        return np.sum(residual**2)

    def gradient(self, X: np.ndarray) -> np.ndarray:
        """
        Calculates the gradient of f(X). ∇f(X) = 2 * P_J(X - U).
        """
        return 2 * self.project_on_omega(X - self.U_dense_observed)

# Core Tools (LMO and Line Search) 

def linear_minimization_oracle(grad: np.ndarray, tau: float) -> np.ndarray:
    """
    Solves the linear minimization subproblem for the nuclear norm ball.
    s_k = argmin_{s: ||s||* <= tau} <s, grad>
    The solution is -tau * u1 * v1.T, where u1, v1 are the top singular vectors of `grad`.
    """
    try:
        u, _, vt = svds(grad, k=1, which='LM')
    except Exception:
        u_full, _, vt_full = np.linalg.svd(grad, full_matrices=False)
        u = u_full[:, 0:1]
        vt = vt_full[0:1, :]
        
    s_k = -tau * (u.reshape(-1, 1) @ vt.reshape(1, -1))
    return s_k

def exact_line_search(d_k: np.ndarray, grad_k: np.ndarray,
                      problem: MatrixCompletionProblem, gamma_max: float = 1.0) -> float:
    """
    Calculates the optimal step-size gamma using the closed-form solution for quadratic objectives.
    """
    proj_d_k = problem.project_on_omega(d_k)
    numerator = -np.sum(grad_k * proj_d_k)
    denominator = 2 * np.sum(proj_d_k**2)

    if denominator < 1e-9:
        return 0.0

    gamma_star = numerator / denominator
    return max(0.0, min(gamma_max, gamma_star))

# Optional/Alternative Tools 

def decaying_step_size(k: int) -> float:
    """
    Returns the classic diminishing step-size.
    """
    return 2 / (k + 2)

def armijo_step_size() -> float:
    """
    Implements the Armijo rule for step-size selection.
    """
    pass

### JACK'S PART
def INDICAZIONE_IMPLEMENTAZIONE_FW(n_rows, n_cols, observed_indices, observed_values,
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
        
        # Compute gradient
        grad = compute_gradient(observed_indices, observed_values, completed_matrix)
        # UNDERSTAND WHY SVD IS USED HERE
        U, S, Vt = svd(-grad, full_matrices=False)
        # UNDERSTAND IF THIS IS EQUIVALENT TO THE DIRECTION (S) (see notes about paper)
        low_rank_update = sigma * np.outer(U[:, 0], Vt[0, :])
        # (also to understand what about the previous Jack project is usable here)


        # HERE we decide which stepsizes to use. armjo is pretty esoteric so maybe something differnt is more sneaky
        if step_size_method == 'diminishing':
            step_size = 2 / (iteration + 2)
        elif step_size_method == 'line_search':
            step_size = line_search(completed_matrix, low_rank_update, observed_indices, observed_values)
        elif step_size_method == 'armijo':
            step_size = armijo_rule(completed_matrix, low_rank_update, observed_indices, observed_values, grad)
        else:
            raise ValueError("Invalid step_size_method. Choose from 'diminishing', 'line_search', or 'armijo'")
        
        # acutal update step of the matrix
        completed_matrix = (1 - step_size) * completed_matrix + step_size * low_rank_update
        
        # compute metrics losses and stuff
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

def plot_losses_and_gaps():
    # and possibly other metrics and evaluations