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


def softmax(x):
    """Compute the softmax function for each row of the input x.
    It is crucial that this function is optimized for speed because
    it will be used frequently in later code. 
    Arguments:
    x -- A D dimensional vector or N x D dimensional numpy matrix.
    Return:
    x -- You are allowed to modify x in-place
    """
    orig_shape = x.shape

    if len(x.shape) > 1:
        # matrix
        tmp = np.max(x, axis=1)
        x -= tmp.reshape((x.shape[0], 1))
        x = np.exp(x)
        tmp = np.sum(x, axis=1)
        x /= tmp.reshape((x.shape[0], 1))
    else:
        # vector
        tmp = np.max(x)
        x -= tmp
        x = np.exp(x)
        tmp = np.sum(x)
        x /= tmp

    assert x.shape == orig_shape
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


def naiveSoftmaxLossAndGradient(
    centerWordVec,
    outsideWordIdx,
    outsideVectors,
    dataset
):
    """ Naive Softmax loss & gradient function for word2vec models
    Implement the naive softmax loss and gradients between a center word's 
    embedding and an outside word's embedding. This will be the building block
    for our word2vec models.
    Arguments:
    centerWordVec -- numpy ndarray, center word's embedding
                    (v_c in the pdf handout)
    outsideWordIdx -- integer, the index of the outside word
                    (o of u_o in the pdf handout)
    outsideVectors -- outside vectors (rows of matrix) for all words in vocab
                      (U (|V| x n) in the pdf handout)
    dataset -- needed for negative sampling, unused here.
    Return:
    loss -- naive softmax loss
    gradCenterVec -- the gradient with respect to the center word vector
                     (dJ / dv_c in the pdf handout)
    gradOutsideVecs -- the gradient with respect to all the outside word vectors
                    (dJ / dU)
    """

    values = np.dot(outsideVectors, centerWordVec)
    y_hat = softmax(values)
    loss = -np.log(y_hat[outsideVectors])

    y_diff = y_hat
    y_diff[outsideWordIdx] -= 1
    gradCenterVec = np.dot(outsideVectors.T, y_diff)
    gradOutsideVecs = np.dot(
        y_diff[:, np.newaxis], outsideVectors[np.newaxis, :])

    return loss, gradCenterVec, gradOutsideVecs


def getNegativeSamples(outsideWordIdx, dataset, K):
    """ Samples K indexes which are not the outsideWordIdx """
    negSampleWordIndices = [None] * K
    for k in range(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == outsideWordIdx:
            newidx = dataset.sampleTokenIdx()
        negSampleWordIndices[k] = newidx
    return negSampleWordIndices


def negSamplingLossAndGradient(
    centerWordVec,
    outsideWordIdx,
    outsideVectors,
    dataset,
    K=10
):
    """ Negative sampling loss function for word2vec models
    Implement the negative sampling loss and gradients for a centerWordVec
    and a outsideWordIdx word vector as a building block for word2vec
    models. K is the number of negative samples to take.
    Note: The same word may be negatively sampled multiple times. For
    example if an outside word is sampled twice, you shall have to
    double count the gradient with respect to this word. Thrice if
    it was sampled three times, and so forth.
    Arguments/Return Specifications: same as naiveSoftmaxLossAndGradient
    """
    negSampleWordIndices = getNegativeSamples(outsideWordIdx, dataset, K)

    o_vector = outsideVectors[outsideWordIdx]
    neg_vectors = outsideVectors[negSampleWordIndices]

    o_value = np.dot(o_vector, centerWordVec)
    neg_values = np.dot(neg_vectors, centerWordVec)

    o_prob = sigmoid(o_value)
    neg_probs = sigmoid(-neg_values)

    loss = -(np.log(o_prob)+np.sum(np.log(neg_probs)))

    gradCenterVec = (o_prob-1)*o_vector + \
        np.sum((1-neg_probs[:, np.newaxis])*neg_vectors, axis=0)

    gradOutsideVecs = np.zeros_like(outsideVectors)
    gradOutsideVecs[outsideWordIdx] = (o_prob-1)*centerWordVec

    for i, neg_index in enumerate(negSampleWordIndices):
        gradOutsideVecs[neg_index] += (1-neg_probs[i])*centerWordVec

    return loss, gradCenterVec, gradOutsideVecs
