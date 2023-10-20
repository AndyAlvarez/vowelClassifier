import adaline1e
import numpy as np
import pandas as pd 
import toolbox as tb
import matplotlib.pyplot as plt

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
        w_epoch, b_epoch, mse_epoch = adaline1e.adaline1e(W_old, b_old, alpha, P, T)

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


def predict(w, b, alpha, test_data, target_matrix):
    
    data_col = np.shape(test_data)[1]
    predictions = [0] * data_col
    hits = 0
    patterns_missed = []

    for i in range(data_col):
        
        # n
        n = np.dot(w, test_data[i]) + b

        # a
        a = tb.hardlims(n)

        predictions[i] = a

        if a == target_matrix[i].item(): 
            hits += 1
        else:
            patterns_missed.append(i+1)
    
    accuracy = (hits / data_col) * 100

    # return accuracy, index_patterns_missed
    print("=" * 30)
    print("Accuracy", accuracy)
    print("Patterns Missed", patterns_missed)
    print("=" * 30)