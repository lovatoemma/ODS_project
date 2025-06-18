import numpy as np
from functions import MatrixCompletionProblem, linear_minimization_oracle, exact_line_search

# CLASSIC FRANK-WOLF (FW) ALGORITHM
def frank_wolfe_solver(problem: MatrixCompletionProblem, tau: float, max_iter: int = 100):
    """
    Solves the Matrix Completion problem with the classic Frank-Wolfe algorithm.
    Reference: Algorithm 2 in FW_survey.pdf 

    Args:
        problem: The MatrixCompletionProblem object.
        tau: The radius of the nuclear norm ball.
        max_iter: Maximum number of iterations.

    Returns:
        X_k: The solution matrix.
        history: A list of objective function values at each iteration.
    """
    print("Starting Frank-Wolfe Solver...")
    # 1. Choose a starting point x0 in C
    X_k = np.zeros((problem.m, problem.n))
    history = []

    for k in range(max_iter):
        # Log objective value
        obj_val = problem.objective_function(X_k)
        history.append(obj_val)
        
        if k % 10 == 0:
            print(f"Iteration {k}, Objective: {obj_val:.4f}")

        # 3. Compute the gradient
        grad_k = problem.gradient(X_k)
        
        # 4. Compute the Frank-Wolfe vertex s_k
        s_k = linear_minimization_oracle(-grad_k, tau)
        
        # 5. Define the search direction
        d_k = s_k - X_k
        
        # Check stopping criterion (FW gap)
        fw_gap = -np.sum(grad_k * d_k)
        if fw_gap < 1e-5:
            print("Convergence reached (FW gap is small).")
            break
            
        # 6. Compute step-size with line search
        gamma_k = exact_line_search(X_k, d_k, grad_k, problem, gamma_max=1.0)
        
        # Update solution
        X_k = X_k + gamma_k * d_k

    print("Frank-Wolfe Solver finished.")
    return X_k, history

# AWAY STEP FRANK-WOLFE (AFW) ALGORITHM
def away_step_frank_wolfe_solver(problem: MatrixCompletionProblem, tau: float, max_iter: int = 100):
    """
    Solves the problem using the Away-Step Frank-Wolfe algorithm.
    Reference: Algorithm 1 in FW_variants.pdf 
    """
    print("Starting Away-Step Frank-Wolfe Solver...")
    
    # 1. Initialize with an atom
    X_k = np.zeros((problem.m, problem.n))
    grad_k_init = problem.gradient(X_k)
    s0 = linear_minimization_oracle(-grad_k_init, tau)
    
    active_set = {0: s0}  # Dict {id: atom}
    weights = {0: 1.0}     # Dict {id: weight}
    X_k = s0
    
    history = []
    atom_id_counter = 1

    for k in range(max_iter):
        obj_val = problem.objective_function(X_k)
        history.append(obj_val)
        
        if k % 10 == 0:
            print(f"Iteration {k}, Objective: {obj_val:.4f}, Active Set Size: {len(active_set)}")

        grad_k = problem.gradient(X_k)
        
        # 3. FW direction
        s_k = linear_minimization_oracle(-grad_k, tau)
        d_fw = s_k - X_k
        g_fw = -np.sum(grad_k * d_fw)
        
        if g_fw < 1e-5:
            print("Convergence reached (FW gap is small).")
            break

        # 4. Away direction
        away_atom_id, v_k = max(active_set.items(), key=lambda item: np.sum(grad_k * item[1]))
        d_away = X_k - v_k
        g_away = -np.sum(grad_k * d_away)
        
        # 6. Choose the best direction
        if g_fw >= g_away:
            # FW step
            direction = d_fw
            gamma_max = 1.0
            is_fw_step = True
        else:
            # Away step
            direction = d_away
            # Max step-size to maintain convexity
            gamma_max = weights[away_atom_id] / (1.0 - weights[away_atom_id]) if weights[away_atom_id] < 1.0 else float('inf')
            is_fw_step = False

        # Compute step-size
        gamma_k = exact_line_search(X_k, direction, grad_k, problem, gamma_max)
        
        # Update solution
        X_k = X_k + gamma_k * direction

        # Update weights and active set
        if is_fw_step:
            new_atom_id = atom_id_counter
            active_set[new_atom_id] = s_k
            if new_atom_id not in weights:
                weights[new_atom_id] = 0.0
            atom_id_counter += 1
            
            for atom_id in list(weights.keys()):
                weights[atom_id] *= (1 - gamma_k)
            weights[new_atom_id] += gamma_k
        else: # Away step
            for atom_id in list(weights.keys()):
                if atom_id == away_atom_id:
                    weights[atom_id] = weights[atom_id] * (1 + gamma_k) - gamma_k
                else:
                    weights[atom_id] *= (1 + gamma_k)

        # Clean active set from zero-weight atoms
        for atom_id in list(weights.keys()):
            if weights[atom_id] < 1e-9:
                del weights[atom_id]
                del active_set[atom_id]
    
    print("Away-Step Frank-Wolfe Solver finished.")
    return X_k, history

# PAIRWISE FRANK-WOLFE (PFW) ALGORITHM
def pairwise_frank_wolfe_solver(problem: MatrixCompletionProblem, tau: float, max_iter: int = 100):
    """
    Solves the problem using the Pairwise Frank-Wolfe algorithm.
    Reference: Algorithm 2 in FW_variants.pdf 
    """
    print("Starting Pairwise Frank-Wolfe Solver...")
    
    # Initialization similar to AFW
    X_k = np.zeros((problem.m, problem.n))
    grad_k_init = problem.gradient(X_k)
    s0 = linear_minimization_oracle(-grad_k_init, tau)
    
    active_set = {0: s0}
    weights = {0: 1.0}
    X_k = s0
    
    history = []
    atom_id_counter = 1

    for k in range(max_iter):
        obj_val = problem.objective_function(X_k)
        history.append(obj_val)
        
        if k % 10 == 0:
            print(f"Iteration {k}, Objective: {obj_val:.4f}, Active Set Size: {len(active_set)}")

        grad_k = problem.gradient(X_k)
        
        # Find FW atom (s_k) and Away atom (v_k)
        s_k = linear_minimization_oracle(-grad_k, tau)
        away_atom_id, v_k = max(active_set.items(), key=lambda item: np.sum(grad_k * item[1]))
        
        # Pairwise (swap) direction 
        d_k = s_k - v_k
        
        fw_gap = -np.sum(grad_k * (s_k - X_k)) # Standard FW gap for checking
        if fw_gap < 1e-5:
            print("Convergence reached (FW gap is small).")
            break
            
        # Max step-size is the weight of the away atom 
        gamma_max = weights[away_atom_id]
        
        gamma_k = exact_line_search(X_k, d_k, grad_k, problem, gamma_max)
        
        # Update solution
        X_k = X_k + gamma_k * d_k
        
        # Update weights: move mass from v_k to s_k
        # Add new atom s_k if it doesn't exist
        
        # A simple way to find if the atom already exists
        found_id = -1
        for a_id, atom in active_set.items():
            if np.allclose(atom, s_k):
                found_id = a_id
                break
        
        if found_id != -1:
            s_k_id = found_id
        else:
            s_k_id = atom_id_counter
            active_set[s_k_id] = s_k
            weights[s_k_id] = 0.0
            atom_id_counter += 1

        weights[away_atom_id] -= gamma_k
        weights[s_k_id] += gamma_k

        # Clean active set
        if weights[away_atom_id] < 1e-9:
            del weights[away_atom_id]
            del active_set[away_atom_id]
                
    print("Pairwise Frank-Wolfe Solver finished.")
    return X_k, history