#
# Word2Vec vectors to distances.
#
# This file contains helper functions for selecting relevant objects from
# arbitrary actions.
#

import numpy as np
from scipy.spatial.distance import cosine

#
# Normalize word2vec vectors.
#
def normalize(a):
    for i in xrange(a.shape[0]):
        a[i] /= np.linalg.norm(a[i])
    return a

#
# Cosine similarity between vectors.
#
def cos_sim(a, b):
    return 1 - cosine(a, b)

#
# Select top n vectors from b given a vector a.
#
def topn(a, b, n):
    cs = np.zeros(b.shape[0])
    for i in xrange(b.shape[0]):
        cs[i] = cos_sim(a, b[i])
    co = np.argsort(cs)[::-1]
    ci = co[:n]
    cw = cs[co[:n]]
    # Normalize the scores to add to one per action, not used in paper.
    #if len(cw) > 0:
    #    cw /= np.sum(cw)
    return ci, cw
