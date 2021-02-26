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


def glove_adagrad_wrapper(gloveModel, wordVectors, wordBiases, cooccurrences):
    batchsize = 50
    loss = 0.0

    gradVectors = np.zeros(wordVectors.shape)
    gradBiases = np.zeros(wordBiases.shape)

    N = wordVectors.shape[0]

    centerVectors = wordVectors[:int(N / 2), :]
    contextVectors = wordVectors[int(N / 2):, :]
    centerBiases = wordBiases[:int(N / 2)]
    contextBiases = wordBiases[int(N / 2):]

    for _ in range(batchsize):
        centerWordId = random.randint(0, N)
        centerWord, contextWord, cooccurrence = cooccurrences[centerWordId]
        l, gradCenterVecs, gradContextVecs, gradCenterBiases, gradContextBiases = glove(
            centerWord, contextWord, cooccurrence, centerVectors, contextVectors, centerBiases, contextBiases)
        loss += l / batchsize

        # Should be adagrade.
        gradVectors[:int(N / 2), :] += gradCenterVecs / batchsize
        gradVectors[int(N / 2):, :] += gradContextVecs / batchsize
        gradBiases[:int(N / 2)] += gradCenterBiases / batchsize
        gradBiases[int(N / 2):] / batchsize

    return loss, gradVectors, gradBiases
