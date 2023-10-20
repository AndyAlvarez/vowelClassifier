'''
percep1e executes 1 full epoch of perceptron learning by calling percep1p repeatedly.

Arguments: W0, b0, P, T -> (initial weight matrix, initial bias vector, pattern matrix, target matrix)
Returns: W_epoch, b_epoch, nze -> (final weight matrix, final bias vector, number of nonzero errors in epoch)
'''

# Imports
import percep1p
import numpy as np

def percep1e(W0, b0, P, T):

    # Weight Matrix Dimension
    #rows_w, cols_w = np.shape(W0) # rows: # of Processing Elements (PEs), cols: output

    # Pattern Matrix Dimension
    #rows_p, cols_p = np.shape(P) # rows: values, cols: # of patterns
    cols_p = np.shape(P)[0]
    # Weights become old weights
    W_old = W0
    b_old = b0

    # Nonzero Error Counter
    nze = 0

    # Loop to update weights for each pattern
    for i in range(cols_p):
        p_i = P[i]
        t_i = T[i]

        W_new, b_new = percep1p.percep1p(W_old, b_old, p_i, t_i)

        # update nze: increment only when no changes have been made to the weights and biases
        if W_new != W_old or b_new != b_old:
            nze += 1

        # New weights and biases become the old ones for the next pattern
        W_old = W_new
        b_old = b_new

    # Set the final weights and biases after finishing an epoch
    W_epoch = W_new
    b_epoch = b_new

    return W_epoch, b_epoch, nze
