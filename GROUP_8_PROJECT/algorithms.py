import numpy as np
from functions import MatrixCompletionProblem, linear_minimization_oracle, exact_line_search, diminishing_step_size, armijo_step_size

###################
### ACTIVE SETS ###
###################

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

#####################################
### FINAL ALGORITHM WITH VARIANTS ###
#####################################
def unified_frank_wolfe_solver(variant: str, stepsize: str, problem: MatrixCompletionProblem, tau: float, max_iter: int = 100):
    """
    Unified solver for different Frank-Wolfe variants.
    
    Args:
        variant: 'classic', 'away_step', or 'pairwise'.
        stepsize: 'exact', 'diminishing' or 'armijo'.
        problem: The MatrixCompletionProblem object.
        tau: The radius of the nuclear norm ball.
        max_iter: Maximum number of iterations.
    
    Returns:
        X_k: The solution matrix.
        history: A list of objective function values at each iteration.
    """
    print(f"Starting Frank-Wolfe Solver... selected {variant} variant")
    
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
        s0 = linear_minimization_oracle(grad_k_init, tau)
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
        s_k = linear_minimization_oracle(grad_k, tau)
        d_fw = s_k - X_k
        fw_gap = -np.sum(grad_k * d_fw)
        
        # Check stopping criterion (FW gap)
        if fw_gap < 1e-5:
            print(f"Convergence reached (FW gap is {fw_gap}).")
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
            gamma_k = exact_line_search(d_k, grad_k, problem, gamma_max)
        elif stepsize == 'diminishing':
            gamma_k = diminishing_step_size(k, gamma_max)
        elif stepsize == 'armijo':
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
