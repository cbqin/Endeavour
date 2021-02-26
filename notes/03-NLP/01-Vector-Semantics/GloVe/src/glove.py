import numpy as np
import random

from scipy import sparse


def gloveLossAndGradient(
    centerWordVec,
    contextWordVec,
    centerBias,
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


def glove(centerWord, contextWord, cooccurrence,
          centerVectors, contextVectors,
          centerBiases, contextBiases):

    loss = 0
    gradCenterVecs = np.zeros(centerVectors.shape)
    gradContextVecs = np.zeros(contextVectors.shape)
    gradCenterBiases = np.zeros(centerBiases.shape)
    gradContextBiases = np.zeros(contextBiases.shape)

    centerVector = centerVectors[centerWord]
    contextVector = contextVectors[contextWord]
    centerBias = centerBiases[centerWord]
    contextBias = contextBiases[contextWord]

    loss, gradCenterVec, gradContextVec, gradCenterBias, gradContextBias =\
        gloveLossAndGradient(centerVector, contextVector,
                             centerBias, contextBias, cooccurrence)

    gradCenterVecs[centerWord] += gradCenterVec
    gradContextVecs[contextWord] += gradContextVec
    gradCenterBiases[centerWord] += gradCenterBias
    gradContextBiases[contextWord] += gradContextBias

    return loss, gradCenterVecs, gradContextVecs, \
        gradCenterBiases, gradContextBiases


def glove_adagrad_wrapper():
    pass
