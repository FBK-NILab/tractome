# -*- coding: utf-8 -*-

"""This module implements the computation of the dissimilarity
representation of a set of objects from a set of prototypes given a
distance function. Various prototype selection policies are available.

Copyright (c) 2012-2014, Emanuele Olivetti and Eleftherios Garyfallidis
Distributed under the BSD 3-clause license. See COPYING.txt.
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.stats import pearsonr as correlation
from sys import stdout


def furthest_first_traversal(S, k, distance, permutation=True):
    """This is the farthest first traversal (fft) algorithm which is
    known to be a good sub-optimal solution to the k-center problem.

    See for example:
    Hochbaum, Dorit S. and Shmoys, David B., A Best Possible Heuristic
    for the k-Center Problem, Mathematics of Operations Research, 1985.

    or: http://en.wikipedia.org/wiki/Metric_k-center
    """
    # do an initial permutation of S, just to be sure that objects in
    # S have no special order. Note that this permutation does not
    # affect the original S.
    if permutation:
        idx = np.random.permutation(S.shape[0])
        S = S[idx]       
    else:
        idx = np.arange(S.shape[0], dtype=np.int)
    T = [0]
    while len(T) < k:
        z = distance(S, S[T]).min(1).argmax()
        T.append(z)
    return idx[T]


def subset_furthest_first(S, k, distance, permutation=True, c=2.0):
    """Stochastic scalable version of the fft algorithm based in a
    random subset of a specific size.

    See: E. Olivetti, T.B. Nguyen, E. Garyfallidis, The Approximation
    of the Dissimilarity Projection, Proceedings of the 2012
    International Workshop on Pattern Recognition in NeuroImaging
    (PRNI), vol., no., pp.85,88, 2-4 July 2012 doi:
    10.1109/PRNI.2012.13

    D. Turnbull and C. Elkan, Fast Recognition of Musical Genres
    Using RBF Networks, IEEE Trans Knowl Data Eng, vol. 2005, no. 4,
    pp. 580-584, 17.
    """
    size = max(1, np.ceil(c * k * np.log(k)))
    if permutation:
        idx = np.random.permutation(S.shape[0])[:size]       
    else:
        idx = range(size)
    # note: no need to add extra permutation here below:
    return idx[furthest_first_traversal(S[idx], k, distance, permutation=False)]


def compute_disimilarity(data, distance, prototype_policy, num_prototypes, verbose=False, size_limit=500000):
    """Compute dissimilarity matrix given data, distance,
    prototype_policy and number of prototypes.
    """
    print "Computing disimilarity data for the original data:",
    data_original = data
    num_proto = num_prototypes
    if data.shape[0] > size_limit:
        print
        print "Datset too big: subsampling to %s entries only!" % size_limit
        data = data[np.random.permutation(data.shape[0])[:size_limit], :]
    
    print prototype_policy    
    print "number of prototypes:", num_proto
    stdout.flush()
    if verbose: print("Generating %s prototypes as" % num_proto),
    # Note that we use the original dataset here, not the subsampled one!
    if prototype_policy=='random':
        if verbose: print("random subset of the initial data.")
        prototype_idx = np.random.permutation(data_original.shape[0])[:num_proto]
        prototype = [data_original[i] for i in prototype_idx]
    elif prototype_policy=='fft':
        prototype_idx = furthest_first_traversal(data_original, num_proto, distance)
        prototype = [data_original[i] for i in prototype_idx]
    elif prototype_policy=='sff':
        prototype_idx = subset_furthest_first(data_original, num_proto, distance)
        prototype = [data_original[i] for i in prototype_idx]                
    else:
        raise Exception                

    if verbose: print("Computing dissimilarity matrix.")
    data_disimilarity = distance(data, prototype)               
                
    print
    return data_disimilarity
