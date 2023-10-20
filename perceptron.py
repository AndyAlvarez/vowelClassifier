'''
Implements the perceptron learning algorithm through as many as a specified number of epochs unless
there are 0 nonzero erros in an epoch less that the specified max epochs.

Arguments: W0, b0, P, T, MAX_EPOCHS
Returns: W_last, b_last, nze_last
'''

# Imports
import percep1e
import toolbox as tb
import numpy as np
import matplotlib.pyplot as plt

def perceptrn(W0, b0, P, T, MAX_EPOCHS):

    W04e = W0 
    b04e = b0

    nze_values = []
    epochs = []

    for i in range(1, MAX_EPOCHS):
        W_epoch, b_epoch, nze = percep1e.percep1e(W04e, b04e, P, T)

        nze_values.append(nze)
        epochs.append(i)

        # Check if this epoch ended with no(zero) patterns having error
        # If no premature break, recirculate the weights and biases after concluding the epoch
        if nze == 0:
            break
        else: 
            W04e = W_epoch
            b04e = b_epoch
    
    # After concluding max epochs, store and return the final results as "last" to later keep track of 
    # weight, bias, and error counter evolution
    W_last = W_epoch
    b_last = b_epoch
    nze_last = nze
    
    # Plots: Shows how many of the 25 training patterns prompted a change of weights
    plt.plot(epochs, nze_values)
    plt.xlabel('Epochs')
    plt.ylabel('Nonzero Errors')
    plt.title('Nonzero Errors vs. Epoch')
    plt.show()

    return W_last, b_last, nze_last



def predict(w, b, test_data, target_mtrx):

    data_col = np.shape(test_data)[1]
    predictions = [0] * data_col
    hits = 0
    patterns_missed = []

    for i in range(data_col):

        # n
        n = np.dot(w, test_data[i]) + b

        # Activation
        a = tb.hardlims(n)
        
        predictions[i] = a

        if a == target_mtrx[i].item(): 
            hits += 1
        else:
            patterns_missed.append(i+1)
    
    accuracy = (hits / data_col) * 100

    # return accuracy, index_patterns_missed
    print("=" * 30)
    print("Accuracy", accuracy)
    print("Patterns Missed", patterns_missed)
    print("=" * 30)
