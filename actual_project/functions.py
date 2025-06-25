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
        Returns a dense numpy array for compatibility with downstream code.
        """
        return 2 * self.project_on_omega(X - self.U_dense_observed)

# Core Tools 
#################
### DIRECTION ###  
#################

# LMO
def linear_minimization_oracle(grad: np.ndarray, tau: float) -> np.ndarray:
    """
    Solves the linear minimization subproblem for the nuclear norm ball.
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

##################
### STEP SIZES ###
##################

# LINE SEARCH
def exact_line_search(d_k: np.ndarray, grad_k: np.ndarray,
                      problem: MatrixCompletionProblem, gamma_max: float = 1.0) -> float:
    """STEPSIZE STRATEGY: Calculates the optimal step-size gamma."""
    proj_d_k = problem.project_on_omega(d_k)
    numerator = -np.sum(grad_k * proj_d_k)
    denominator = 2 * np.sum(proj_d_k**2)

    if denominator < 1e-9:
        return 0.0

    gamma_star = numerator / denominator
    return max(0.0, min(gamma_max, gamma_star))

# DIMINISHING
def diminishing_step_size(k: int, gamma_max: float = 1.0) -> float:
    """STEPSIZE STRATEGY: Returns the classic diminishing step-size gamma = 2/(k+2)."""
    # We still clip by gamma_max, which is crucial for Away and Pairwise steps.
    return min(gamma_max, 2 / (k + 2))
# Ciao Rebe

# ARMIJO   
def armijo_step_size(problem, X_k: np.ndarray, d_k: np.ndarray, grad_k: np.ndarray,
                               gamma_max: float = 1.0, c: float = 1e-4, beta: float = 0.8) -> float:
    """
    Performs Armijo backtracking line search to find an acceptable step-size.
    This implementation is based on the description in FW_survey.pdf, Section 4.

    Args:
        problem: The MatrixCompletionProblem object, which provides the objective function.
        X_k: The current iterate matrix.
        d_k: The current search direction.
        grad_k: The gradient at X_k.
        gamma_max: The initial (and maximum) step-size to try. Corresponds to alpha_k^max in the paper.
        c: The sufficient decrease parameter. Corresponds to gamma in the paper.
        beta: The backtracking factor for shrinking the step-size. Corresponds to delta in the paper.

    Returns:
        An acceptable step-size gamma_k.
    """
    gamma_k = gamma_max
    
    # Calculate the objective value at the current point
    obj_k = problem.objective_function(X_k)
    
    # Calculate the directional derivative, which should be negative for a descent direction
    directional_derivative = np.sum(grad_k * d_k)
    
    # Iteratively shrink gamma until the Armijo condition is met
    while True:
        # Calculate the objective value at the new trial point
        obj_new = problem.objective_function(X_k + gamma_k * d_k)
        
        # The Armijo sufficient decrease condition
        armijo_condition = obj_k + c * gamma_k * directional_derivative
        
        if obj_new <= armijo_condition:
            # If the condition is satisfied, we have found a good step-size
            return gamma_k
        
        # If not, shrink the step-size by the backtracking factor
        gamma_k *= beta
        
        # Failsafe to prevent infinitely small step-sizes
        if gamma_k < 1e-12:
            return 0.0
        
def accuracy_spectral(problem: MatrixCompletionProblem, X: np.ndarray) -> float:
    """
    Computes the spectral accuracy of the current solution X.
    This is defined as the largest singular value of the residual matrix.
    """
    residual = problem.project_on_omega(X - problem.U_dense_observed)
    u, s, vt = np.linalg.svd(residual, full_matrices=False)
    return s[0]  # Return the largest singular
