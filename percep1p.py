'''
percep1p learns and updates the old weights and bias vectors from 1 pattern p and its target.

Arguments: W_old, b_old, p, t -> (old weights, old bias, pattern, target)
Returns: W_new, b_new -> (new weights, new bias)
'''

# Imports
import numpy as np
import toolbox as tb

def percep1p(W_old, b_old, p, t):
    
    n = np.dot(W_old, p) + b_old

    # Activation
    a = tb.hardlims(n)

    #Finding Error
    e = t.item() - a

    # Updating Weights
    W_new = W_old + e * (np.transpose(p))
    b_new = b_old + e

    return W_new.tolist(), b_new
