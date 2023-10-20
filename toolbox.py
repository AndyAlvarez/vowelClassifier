import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def hardlim(n):

    if n >= 0:
        return 1
    else:
        return 0
    

def hardlims(n):

    if n >= 0:
        return 1
    else:
        return -1

'''
Converts a an input pattern (20x1) column vector p to a 4x5 matrix
'''
def col2mtx(p):

    PM = np.reshape(p, (4, 5), order='F')
    return PM

'''
Display an input pattern given as a matrix PM
'''
def dispapm(PM):

    mtx = np.flipud(PM)
    r, c = mtx.shape
    ckb = np.zeros((r, c))
    ckb[:r, :c] = mtx

    cmap = plt.cm.gray
    reversed_cmap = cmap.reversed()

    plt.pcolor(ckb, cmap=reversed_cmap)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.colorbar()
    plt.show()

'''
Toggles a pixel in a 4x5 matrix - WHICH HAS ALREADY BEEN CONVERTED TO A COLUMN VECTOR "colin"
'''
def toggleP1matlab(col_in, x, y):
    indx = ((x - 1) * 4) + y
    old_val = col_in[indx]
    col_out = col_in

    col_out[indx] = old_val * -1

def toggleP1(col_in, x, y):
    indx = (x * 4) - y
    old_val = col_in[indx]
    col_out = col_in

    col_out[indx] = old_val * -1

'''
Turns on a pixel in a 4x5 matrix 
'''
def turnOnP1():
    pass