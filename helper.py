#
# Helper functions for boxes.
#
# Categories of helper functions:
# 1. Overlap functions.
# 2. Spatial-aware object embedding functions.
# 3. Misc.
#

import numpy as np
from scipy.stats import entropy

from sklearn.metrics import average_precision_score

#
# Helper functions category 1: Overlap functions.
#

#
# 1.1
# Compute the intersection between two boxes.
#
def boxintersect(a,b):
    if a[0] > b[2] or b[0] > a[2] or a[1] > b[3] or b[1] > a[3]:
        return 0
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h

#
# 1.2
# Overlap between two boxes.
#
def box_overlap(a, b):
    if a[2] > b[0] and b[2] > a[0] and a[3] > b[1] and b[3] > a[1]:
        i  = min(a[2],b[2]) - max(a[0],b[0])
        i *= min(a[3],b[3]) - max(a[1],b[1])
        i  = float(i)
        a1 = ((a[2]-a[0]) * (a[3]-a[1]))
        a2 = ((b[2]-b[0]) * (b[3]-b[1]))
        return i / (a1 + a2 - i)
    return 0.

#
# 1.3
# Overlap between box and list of other boxes.
#
def liou(a, b):
    iou = np.zeros(b.shape[0])
    for i in xrange(b.shape[0]):
        iou[i] = box_overlap(a, b[i])
    return iou

#
# 1.4
# Compute the intersection-over-union score for two proposals from the
# same video.
# Optional parameter: ss (stride for first tube, in case of sparse annotation
# for second tube.
#
def tube_iou(p1, p2, ss=1):
    # Frame indices.
    p2idxs   = np.where(p2[:,2] >= 0)[0]
    p1       = p1[::ss,:]
    p1f, p2f = p1[:,0].astype(int), p2[:,0].astype(int)
    p2f      = p2f[p2idxs]
    p2       = p2[p2idxs,:]

    # Determine union of frame span.
    tmin = min(np.min(p1f), np.min(p2f))
    tmax = max(np.max(p1f), np.max(p2f))

    # Initialize the overlap scores across frame span.
    span = np.arange(tmin, tmax+1, ss)
    overlaps  = np.zeros(len(span), dtype=np.float)

    # Go through the frame span.
    for d in xrange(len(span)):
        i = span[d]
        p1i, p2i = np.where(p1f == i)[0], np.where(p2f == i)[0]
        # Compute the overlap if frame in both proposals.
        if len(p1i) == 1 and len(p2i) == 1:
            a,b = p1[p1i[0],1:], p2[p2i[0],1:]
            a   = [min(a[0],a[2]), min(a[1],a[3]), max(a[0],a[2]), max(a[1],a[3])]
            b   = [min(b[0],b[2]), min(b[1],b[3]), max(b[0],b[2]), max(b[1],b[3])]
            # Only compute overlap if there is any
            if a[2] > b[0] and b[2] > a[0] and a[3] > b[1] and b[3] > a[1]:
                intersection  = (min(a[2],b[2]) - max(a[0],b[0]))
                intersection *= (min(a[3],b[3]) - max(a[1],b[1]))
                intersection  = float(intersection)
                area1         = ((a[2]-a[0]) * (a[3]-a[1]))
                area2         = ((b[2]-b[0]) * (b[3]-b[1]))
                overlaps[d] = intersection / (area1 + area2 - intersection)

    # Return the mean overlap over the frame span
    return np.mean(overlaps)

#
# Helper functions category 2: Embedding functions.
#

#
# 2.1
# Minimal edge distance between two boxes.
#
def box_dist(a, b):
    if boxintersect(a,b) > 0:
        return 0
    ae = np.array([[a[0],a[1]], [a[2],a[1]], [a[0],a[3]], [a[2],a[3]]])
    be = np.array([[b[0],b[1]], [b[2],b[1]], [b[0],b[3]], [b[2],b[3]]])
    mind = np.min(np.linalg.norm(ae-be[0], axis=1))
    for i in xrange(1, be.shape[0]):
        nd   = np.min(np.linalg.norm(ae-be[i], axis=1))
        mind = min(mind, nd)
    return mind

#
# 2.2
# Tile distributions with 9 tiles. a=person, b=object.
#
def tiledist(a, b):
    d = np.zeros(9)
    e = 1e6
    # Above left.
    d[0] = boxintersect([0, 0, a[0], a[1]], b)
    # Above center.
    d[1] = boxintersect([a[0], 0, a[2], a[1]], b)
    # Above right.
    d[2] = boxintersect([a[2], 0, e, a[1]], b)
    # Left.
    d[3] = boxintersect([0, a[1], a[0], a[3]], b)
    # On.
    d[4] = boxintersect(a, b)
    # Right.
    d[5] = boxintersect([a[2], a[1], e, a[3]], b)
    # Below left.
    d[6] = boxintersect([0, a[3], a[0], e], b)
    # Below center.
    d[7] = boxintersect([a[0], a[3], a[2], e], b)
    # Below right.
    d[8] = boxintersect([a[2], a[3], e, e], b)
    return d / float((b[2] - b[0]) * (b[3] - b[1]))

#
# 2.3
# Find pairs of high scoring and high overlapping boxes for Viterbi.
#
def viterbi_scores(b1, s1, b2, s2, iouw):
    scores = np.zeros((s1.shape[0], s2.shape[0]))
    for i in xrange(s1.shape[0]):
        iou = liou(b1[i], b2)
        scores[i] = s1[i] + s2 + iouw * iou
    return scores

#
# Helper functions category 3: Misc functions.
#
    
#
# 3.1
# Jensen-Shannon Divergence.
#
def jensen_shannon_divergence(p, q):
    apq = 0.5 * (p + q)
    return 0.5 * entropy(p, apq, 2) + 0.5 * entropy(q, apq, 2)

#
# 3.2
# Remove elements from a tube that correspond to non-annotations.
# Used for experiments on Hollywood2Tubes, where the lack of action is
# annotated with -1 values for the coordindates in the frame.
#
def tube_trim(tube):
    keep = np.where(tube[:,1] >= 0)[0]
    return tube[keep,:]
    
#
# 3.3
# Interpolate a tube (done e.g. for UCF-101 due to its many videos).
#
def tube_interpolate(tube, scores, stride, nr_frames):
    if tube.shape[0] == nr_frames:
        return tube
    
    ntube   = np.zeros((nr_frames, tube.shape[1]), dtype=tube.dtype)
    nscores = np.zeros(nr_frames)

    for i in xrange(nr_frames):
        i1, i2 = i / stride, i / stride + 1
        w      = (i % stride) / float(stride)

        ntube[i,0] = i
        if i2 < tube.shape[0]:
            ntube[i,1] = (1-w) * tube[i1,1] + w * tube[i2,1]
            ntube[i,2] = (1-w) * tube[i1,2] + w * tube[i2,2]
            ntube[i,3] = (1-w) * tube[i1,3] + w * tube[i2,3]
            ntube[i,4] = (1-w) * tube[i1,4] + w * tube[i2,4]
            nscores[i] = (1-w) * scores[i1] + w * nscores[i2]
        else:
            ntube[i,1] = tube[i1,1]
            ntube[i,2] = tube[i1,2]
            ntube[i,3] = tube[i1,3]
            ntube[i,4] = tube[i1,4]
            nscores[i] = scores[i1]

    return ntube, nscores

