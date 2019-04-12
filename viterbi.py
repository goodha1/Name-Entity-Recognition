import numpy as np
from numpy import array
from numpy import random
import sys
def run_viterbi(emission_scores, trans_scores, start_scores, end_scores):
    """Run the Viterbi algorithm.

    N - number of tokens (length of sentence)
    L - number of labels

    As an input, you are given:
    - Emission scores, as an NxL array
    - Transition scores (Yp -> Yc), as an LxL array
    - Start transition scores (S -> Y), as an Lx1 array
    - End transition scores (Y -> E), as an Lx1 array

    You have to return a tuple (s,y), where:
    - s is the score of the best sequence
    - y is a size N array of integers representing the best sequence.
    """
    L = start_scores.shape[0]
    assert end_scores.shape[0] == L
    assert trans_scores.shape[0] == L
    assert trans_scores.shape[1] == L
    assert emission_scores.shape[1] == L
    N = emission_scores.shape[0]

    Start = start_scores.shape
    Emission = emission_scores.shape
    Viterbi = np.zeros(shape = (N,L))
    Backpointer = np.zeros(shape =(N,L), dtype = int)
    # Viterbi.fill(-sys.maxint -1)
    y = []

    for i in xrange(L):
        Viterbi[0][i] = start_scores[i] + emission_scores[0][i]
        Backpointer[0][i] = -1
    for i in xrange(1,N):
        for m in xrange(L):
            Viterbi[i][m] = -sys.maxint -1
            for n in xrange(L):
                tmp = Viterbi[i-1][n] + trans_scores[n][m]
                if tmp - Viterbi[i][m] > 0:
                    Viterbi[i][m] = tmp
                    Backpointer[i][m] = n
            Viterbi[i][m] +=  emission_scores[i][m]

    maxp = 0
    maxv = -sys.maxint -1
    for i in xrange(L):
        if Viterbi[N-1][i] + end_scores[i] > maxv:
            maxp = i
            maxv = Viterbi[N-1][i] + end_scores[i]
    y.append(maxp)
    for i in xrange(1,N):
        y.append(Backpointer[N - i][maxp])
        maxp = Backpointer[N - i][maxp]
    y.reverse()

    return (maxv, y)


    # for i in xrange(N):
    #     # stupid sequence
    #     y.append(i % L)
    # # score set to 0
    # return (0.0, y)
