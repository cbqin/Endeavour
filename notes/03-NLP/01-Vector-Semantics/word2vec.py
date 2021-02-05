import numpy as np
import random


def normalizeRows(x):
    """ Row normalization function
    Implement a function that normalizes each row of a matrix to have
    unit length.
    """
    N = x.shape[0]
    x /= np.sum(np.sqrt(x), axis=1).reshape((N, 1))+1e-30
    return x


def sigmoid(x):
    """
    Compute the sigmoid function for the input here.
    Arguments:
    x -- A scalar or numpy array.
    Return:
    s -- sigmoid(x)
    """

    s = 1.0/(1+np.exp(-x))

    return s
