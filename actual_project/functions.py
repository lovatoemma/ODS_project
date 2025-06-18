def initialize_matrix(n_rows, n_cols, rank, observed_matrix):

def compute_loss(observed_indices, observed_values, completed_matrix):

def compute_gradient(observed_indices, observed_values, completed_matrix):

def line_search(completed_matrix, low_rank_update, observed_indices, observed_values, num_steps=1000):
# and possibly other stepsizes variants

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