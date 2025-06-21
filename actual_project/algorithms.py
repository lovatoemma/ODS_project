import numpy as np
from functions import MatrixCompletionProblem, linear_minimization_oracle, exact_line_search, decaying_step_size, armijo_step_size

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


# since there are a lot of common steps, we can handle all three algorithms in a unified function
# we can help ourself with supporting functions to handle the differences in the algorithms
# NOTE: it is still not completely clear to me the exact handling of the active set for the two variants
def _update_active_set_away_step(
    s_k, active_set, weights, away_atom_id, gamma_k,
    is_fw_step, atom_id_counter
):
    # if we do a FW step, we need to add the new atom s_k to the active set
    if is_fw_step:
        # we assign to the active set, for the new_atom_id, the atom s_k
        new_atom_id = atom_id_counter
        active_set[new_atom_id] = s_k
        # If the new atom is not already in the weights, initialize it
        if new_atom_id not in weights:
            weights[new_atom_id] = 0.0
        # Increment the atom ID counter
        atom_id_counter += 1
        # Update weights: move mass from away_atom_id to new_atom_id
        for atom_id in list(weights.keys()):
            weights[atom_id] *= (1 - gamma_k)
        weights[new_atom_id] += gamma_k
    else:  # Away step
        # Update weights: move mass from away_atom_id to s_k
        for atom_id in list(weights.keys()):
            if atom_id == away_atom_id:
                weights[atom_id] = weights[atom_id] * (1 + gamma_k) - gamma_k
            else:
                weights[atom_id] *= (1 + gamma_k)

    # Clean up: remove atoms with negligible weights
    to_remove = [atom_id for atom_id, w in weights.items() if w < 1e-9]
    # Remove atoms with negligible weights from both active_set and weights
    for atom_id in to_remove:
        del weights[atom_id]
        del active_set[atom_id]

    return atom_id_counter

def _update_active_set_pairwise(
    s_k, active_set, weights, away_atom_id, gamma_k, atom_id_counter
):
    # Check if s_k already in set
    found_id = None
    for a_id, atom in active_set.items():
        if np.allclose(atom, s_k):
            found_id = a_id
            break
    # If s_k is not in the active set, we add it

    if found_id is None:
        found_id = atom_id_counter
        active_set[found_id] = s_k
        # initialize the new atom weight
        weights[found_id] = 0.0
        atom_id_counter += 1
    # Update weights: move mass from away_atom_id to found_id
    weights[away_atom_id] -= gamma_k
    weights[found_id] += gamma_k

    # Clean up: remove atoms with negligible weights
    if weights[away_atom_id] < 1e-9:
        del weights[away_atom_id]
        del active_set[away_atom_id]

    return atom_id_counter

def unified_frank_wolfe_solver(variant: str, stepsize: str, problem: MatrixCompletionProblem, tau: float, max_iter: int = 100):
    """
    Unified solver for different Frank-Wolfe variants.

    
    Args:
        variant: 'classic', 'away_step', or 'pairwise'.
        stepsize: 'exact', 'decaying' or 'armijo'.
        problem: The MatrixCompletionProblem object.
        tau: The radius of the nuclear norm ball.
        max_iter: Maximum number of iterations.
    
    Returns:
        X_k: The solution matrix.
        history: A list of objective function values at each iteration.
    """
    print(f"Starting Pairwise Frank-Wolfe Solver... selected {variant} variant")
    
    # initialization
    if variant not in ['classic', 'away_step', 'pairwise']:
        raise ValueError("Invalid variant. Choose from 'classic', 'away_step', or 'pairwise'.")
    
    # we start from a null matrix
    X_k = np.zeros((problem.m, problem.n))
    history = []

    if variant != 'classic':
        # For away-step and pairwise variants, we initialize with an atom. This is necessary to have an initial point in the active set.
        # Compute the initial gradient
        grad_k_init = problem.gradient(X_k)
        # Compute the initial FW atom (s0)
        s0 = linear_minimization_oracle(-grad_k_init, tau)
        # Initialize active set and weights
        active_set = {0: s0}
        weights = {0: 1.0}
        # Set the initial solution to the first atom
        X_k = s0
        atom_id_counter = 1    

    for k in range(max_iter):        
        # Log objective value
        obj_val = problem.objective_function(X_k)
        history.append(obj_val)
        
        # Print progress every 10 iterations
        if k % 10 == 0:
            print(f"Iteration {k}, Objective: {obj_val:.4f}")

        # Compute the gradient
        grad_k = problem.gradient(X_k)

        # FW direction
        s_k = linear_minimization_oracle(-grad_k, tau)
        d_fw = s_k - X_k
        fw_gap = -np.sum(grad_k * d_fw)
        
        # Check stopping criterion (FW gap)
        if fw_gap < 1e-5:
            print("Convergence reached (FW gap is small).")
            break

        if variant == 'classic':
            # direction is set to the FW direction
            d_k = d_fw
            gamma_max = 1.0

        else:
            # For away-step and pairwise variants, we need to compute the away direction
            # Find away atom (v_k)
            away_atom_id, v_k = max(active_set.items(), key=lambda item: np.sum(grad_k * item[1]))
            d_away = X_k - v_k
            g_away = -np.sum(grad_k * d_away)

            if variant == 'away_step':
                # Choose the best direction 
                if g_away > fw_gap:
                    # Away step
                    d_k = d_away
                    # Max step-size to maintain convexity
                    gamma_max = weights[away_atom_id] / (1.0 - weights[away_atom_id]) if weights[away_atom_id] < 1.0 else float('inf')
                    # Away step flag, used later to update the active set
                    is_fw_step = False   
                else:
                    # FW step
                    d_k = d_fw
                    gamma_max = 1.0
                    # FW step flag, used later to update the active set
                    is_fw_step = True
    
            elif variant == 'pairwise':
                # Pairwise direction
                d_k = s_k - v_k
                # Max step-size is the weight of the away atom
                gamma_max = weights[away_atom_id]

        # Compute step-size with line search & Update solution
        # TODO : experiment with different step-size strategies
        # 
        if stepsize == 'exact':
            gamma_k = exact_line_search(X_k, d_k, grad_k, problem, gamma_max)
        elif stepsize == 'decaying':
            gamma_k = decaying_step_size(k, max_iter, gamma_max)
        elif stepsize == 'armijo':
            # TODO: TO BE IMPLEMENTED!!!! (if we want)
            gamma_k = armijo_step_size() 
        # Update solution
        X_k = X_k + gamma_k * d_k

        # Update weights and active set
        # this is necessary for away-step and pairwise variants since they need to mantain the active set
        # classic FW does not need this step since it moves always to a new atom
        if variant == 'away_step':

            atom_id_counter = _update_active_set_away_step(
                s_k, active_set, weights, away_atom_id, gamma_k,
                is_fw_step, atom_id_counter
            )
        elif variant == 'pairwise':
            atom_id_counter = _update_active_set_pairwise(
                s_k, active_set, weights, away_atom_id, gamma_k, atom_id_counter
            )

    print("Frank-Wolfe Solver finished.")
    return X_k, history