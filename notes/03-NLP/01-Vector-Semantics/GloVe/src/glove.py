import numpy as np
import random

from scipy import sparse


def lossAndGradient(
    centerWordVec,
    centerBias,
    contextWordVec,
    contextBias,
    cooccurrence,
    xMax=100,
    alpha=0.75
):
    weight = (cooccurrence / xMax)**alpha if cooccurrence < xMax else 1

    innerLoss = np.dot(centerWordVec, contextWordVec) + \
        centerBias + contextBias + np.log(cooccurrence)
    loss = weight * innerLoss**2

    gradCenterVec = weight * innerLoss * contextWordVec
    gradContextVec = weight * innerLoss * centerWordVec

    gradCenterBias = weight * innerLoss
    gradContextBias = weight * innerLoss

    return loss, gradCenterVec, gradContextVec, gradCenterBias, gradContextBias


def glove(currentCenterWord, word2Ind,
          centerWordVectors, outsideVectors):
    pass


def glove_adagrad_wrapper():
    pass
