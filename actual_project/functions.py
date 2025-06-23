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
# LMO
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

# LINE SEARCH
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

# DIMINISHING
def decaying_step_size(k: int, gamma_max: float = 1.0) -> float:
    """Returns the classic diminishing step-size gamma = 2/(k+2), clipped by gamma_max."""
    return min(gamma_max, 2 / (k + 2))
