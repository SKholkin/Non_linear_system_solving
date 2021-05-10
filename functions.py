import numpy as np

def analytical_jacobian(x):
    j11 = 1
    j12 = -np.sin(x[1] - 1)
    j21 = np.sin(x[0])
    j22 = 1
    return [[j11, j12], [j21, j22]]

def f1(x1, x2):
    return np.cos(x2 - 1) + x1 - 0.8

def f2(x1, x2):
    return x2 -  np.cos(x1) - 2

starting_point = [1.5, 4]
