import numpy as np
from scipy import sparse
from functools import wraps
from collections import Counter


def listify(fn):
    """
    Use this decorator on a generator function to make it return a list
    instead.
    """
    @wraps(fn)
    def listified(*args, **kwargs):
        return list(fn(*args, **kwargs))

    return listified


def buildVocab(dataset):

    vocab = Counter()

    for sentence in dataset:
        tokens = sentence.strip().split()
        vocab.update(tokens)

    return {word: (i, freq) for i, (word, freq) in enumerate(vocab.items())}


@listify
def buildCooccur(vocab, dataset, windowSize=10, minCount=0):
    """Build a word-word co-occurrence matrix for given dataset.
    Arguments:
    vocab -- vocab build for corpus, a dictionary, item is like 
             word: (id, freq)
    dataset -- needed for building co-occurrence
    windowSize -- integer, context window size
    minCount -- integer or none, if not none, cooccurrence pairs where either word
    occurs in the corpus fewer than `minCount` times are ignored

    Return:
    a tuple generator, where each element (representing a cooccurrence pair) is of the form
        (center, context, cooccurrence)
    """

    vocabSize = len(vocab)
    id2word = dict((i, word) for word, (i, _) in vocab.items())

    cooccurrences = sparse.lil_matrix((vocabSize, vocabSize), dtype=np.float64)

    for _, sentence in enumerate(dataset):
        tokens = sentence.strip().split()
        tokenIds = [vocab[token][0] for token in tokens]

        for centerI, centerId in enumerate(tokenIds):
            contextIds = [max(0, centerI - windowSize), centerI]
            contextLen = len(contextIds)

            for leftI, leftId in enumerate(contextIds):
                distance = contextLen - leftI
                increment = 1.0 / distance
                cooccurrences[centerId, leftId] += increment
                cooccurrences[leftId, centerId] += increment

    for i, (row, data) in enumerate(zip(cooccurrences.rows,
                                        cooccurrences.data)):
        if vocab[id2word[i]][1] < minCount:
            continue

        for dataIdx, j in enumerate(row):
            if vocab[id2word[j]][1] < minCount:
                continue

            yield i, j, data[dataIdx]
