import numpy as np
import random


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
