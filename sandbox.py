import numpy as np
import pandas as pd
import toolbox as tb
import matplotlib.pyplot as plt
# import percep1p

PP = pd.read_csv(r'datasets/PPfile.csv', header = None)
TA = pd.read_csv(r'datasets/TAfile.csv', header = None)

MAX_EPOCHS = 40

# Initial Weights and bias
W0 = [0] * len(PP[0])
b_init = 0

def ada(W0, b0, alpha, P, T, MAX_EPOCHS):
    W_old = W0
    b_old = b0

    w1_epoch_i = []
    w2_epoch_i = []
    w3_epoch_i = []
    w4_epoch_i = []
    b_epoch_i = []
    MSE_per_epoch = [0] * MAX_EPOCHS
    
    for i in range(MAX_EPOCHS):
        w_epoch, b_epoch, mse_epoch = adaline1e(W_old, b_old, alpha, P, T)

        MSE_per_epoch[i] = mse_epoch

        W_old = w_epoch
        b_old = b_epoch
        last_mse = mse_epoch
    
    # INSTRUCTIONS: Display the final value of the bias and 4 of the 20 weights at the end of each epoch for each pattern
        w1_epoch_i.append(w_epoch[0])
        w2_epoch_i.append(w_epoch[1])
        w3_epoch_i.append(w_epoch[2])
        w4_epoch_i.append(w_epoch[3])
        b_epoch_i.append(b_epoch)

    print("Size of w1 array", np.shape(w1_epoch_i))
    print(w1_epoch_i)
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    # Plotting the MSE per Epoch (Learning Curve)
    axs[0].plot(MSE_per_epoch)
    axs[0].set_title(f'MSE v. Epoch Learning Curve: Alpa = {alpha}')
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("MSE")
    axs[0].set_xlim(0, MAX_EPOCHS)
    axs[0].set_ylim(0, np.max(MSE_per_epoch))

    axs[1].set_xlabel("Epochs")
    # Plotting w1
    (markers, stemlines, baseline) = axs[1].stem(w1_epoch_i)
    plt.setp(stemlines, linestyle='-', color='cyan', linewidth=0 )
    plt.setp(markers, markersize=5, color='cyan', linestyle='-')
    plt.setp(baseline, visible=False)

    # Plotting w2
    (markers, stemlines, baseline) = axs[1].stem(w2_epoch_i)
    plt.setp(stemlines, linestyle='-', color='olive', linewidth=0 )
    plt.setp(markers, markersize=5, color='olive', linestyle='-')
    plt.setp(baseline, visible=False)

    # Plotting w3
    (markers, stemlines, baseline) = axs[1].stem(w3_epoch_i)
    plt.setp(stemlines, linestyle='-', color='green', linewidth=0 )
    plt.setp(markers, markersize=5, color='green', linestyle='-')
    plt.setp(baseline, visible=False)

    # Plotting w4
    (markers, stemlines, baseline) = axs[1].stem(w4_epoch_i)
    plt.setp(stemlines, linestyle='-', color='red', linewidth=0 )
    plt.setp(markers, markersize=5, color='red', linestyle='-')
    plt.setp(baseline, visible=False)

    # Plotting Bias
    (markers, stemlines, baseline) = axs[1].stem(b_epoch_i)
    plt.setp(stemlines, linestyle='-', color='brown', linewidth=0 )
    plt.setp(markers, markersize=5, color='brown', linestyle='-')
    plt.setp(baseline, visible=False)

    axs[1].set_title("4 Weights and Bias v Epoch")
    axs[1].legend(['w1', 'w2', 'w3', 'w4', 'bias'])

    plt.show()

    return W_old, b_old, last_mse



def adaline1e(W_old, b_old, alpha , P, T):

    # Number of Patterns in training set
    p_cols = np.shape(P)[1]

    mse_acc = 0

    for i in range(p_cols):
        w_new, b_new, msek = adaline1p(W_old, b_old, alpha, P[i], T[i])

        W_old = w_new
        b_old = b_new
        mse_acc += msek
    
    mse_epoch = mse_acc / p_cols
    W_epoch = W_old
    b_epoch = b_old

    return W_epoch, b_epoch, mse_epoch


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


W_final_a, b_final_a, mse_final_a = ada(W0, b_init, 0.01, PP, TA, MAX_EPOCHS)

print(W_final_a)
print(b_final_a)
print(mse_final_a)