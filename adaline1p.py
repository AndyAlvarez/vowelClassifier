import numpy as np

def adaline1p(W_old, b_old, alpha, p, t):
    
    # Net Inputs: Linear n = a
    a = np.dot(W_old, p) + b_old

    # Error
    e = t.item() -  a

    # Mean Squared Error
    msek = (np.dot(np.transpose(e), e))

    w_new = W_old + (2 * alpha * np.dot(e, np.transpose(p)))
    b_new = b_old + (2 * alpha * e)

    return w_new, b_new, msek