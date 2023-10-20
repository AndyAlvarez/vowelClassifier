import adaline1p
import numpy as np

def adaline1e(W_old, b_old, alpha , P, T):

    # Number of Patterns in training set
    p_cols = np.shape(P)[1]

    mse_acc = 0

    for i in range(p_cols):
        w_new, b_new, msek = adaline1p.adaline1p(W_old, b_old, alpha, P[i], T[i])

        W_old = w_new
        b_old = b_new
        mse_acc += msek
    
    mse_epoch = mse_acc / p_cols
    W_epoch = W_old
    b_epoch = b_old

    return W_epoch, b_epoch, mse_epoch