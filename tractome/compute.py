import json
import logging
import tempfile

from dipy.utils.optpkg import optional_package
import numpy as np
from sklearn.cluster import MiniBatchKMeans

ray, has_ray, _ = optional_package("ray")


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
        idx = np.arange(S.shape[0], dtype=np.int32)
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
    size = int(max(1, np.ceil(c * k * np.log(k))))
    if permutation:
        idx = np.random.permutation(S.shape[0])[:size]
    else:
        idx = range(size)
    # note: no need to add extra permutation here below:
    return idx[furthest_first_traversal(S[idx], k, distance, permutation=False)]


def compute_dissimilarity(
    data,
    distance,
    prototype_policy,
    num_prototypes,
    verbose=False,
    size_limit=5000000,
    n_jobs=6,
):
    """Compute dissimilarity matrix given data, distance,
    prototype_policy and number of prototypes.
    """
    logging.info("Computing dissimilarity matrix.")
    data_original = data
    num_proto = num_prototypes
    if data.shape[0] > size_limit:
        logging.info("Datset too big: subsampling to %s entries only!" % size_limit)
        data = data[np.random.permutation(data.shape[0])[:size_limit], :]

    logging.info("Number of prototypes: %s" % num_proto)
    if verbose:
        logging.info("Generating %s prototypes as %s" % (num_proto, prototype_policy))
    # Note that we use the original dataset here, not the subsampled one!
    if prototype_policy == "random":
        if verbose:
            logging.info("Random subset of the initial data.")
        prototype_idx = np.random.permutation(data_original.shape[0])[:num_proto]
        prototype = [data_original[i] for i in prototype_idx]
    elif prototype_policy == "fft":
        prototype_idx = furthest_first_traversal(data_original, num_proto, distance)
        prototype = [data_original[i] for i in prototype_idx]
    elif prototype_policy == "sff":
        prototype_idx = subset_furthest_first(data_original, num_proto, distance)
        prototype = [data_original[i] for i in prototype_idx]
    else:
        raise Exception("Unknown prototype policy: %s" % prototype_policy)

    if verbose:
        logging.info("Computing dissimilarity matrix.")
    if has_ray and n_jobs > 1:
        logging.info(
            "Parallel computation of the dissimilarity matrix: %s cpus." % n_jobs
        )

        tmp = np.linspace(0, data.shape[0], n_jobs).astype(np.int32)
        chunks = zip(tmp[:-1], tmp[1:])

        tmp_dir = tempfile.TemporaryDirectory()

        if not ray.is_initialized():
            ray.init(
                _system_config={
                    "object_spilling_config": json.dumps(
                        {
                            "type": "filesystem",
                            "params": {"directory_path": tmp_dir.name},
                        }
                    )
                }
            )

        func = ray.remote(distance)
        func_refs = [func.remote(data[start:end], prototype) for start, end in chunks]

        data_dissimilarity = []
        for i in range(len(func_refs)):
            data_dissimilarity.extend(ray.get(func_refs[i]))

    else:
        data_dissimilarity = distance(data, prototype)

    return data_dissimilarity


def mkbm_clustering(dissimilarity_matrix, n_clusters, streamline_ids):
    """Perform MKBM clustering on the dissimilarity matrix.

    Parameters
    ----------
    dissimilarity_matrix : ndarray
        The dissimilarity matrix to cluster.
    n_clusters : int
        The number of clusters to create.
    streamline_ids : ndarray
        The IDs of the streamlines to cluster.

    Returns
    -------
    dict
        A dictionary mapping cluster centers to lists of streamline IDs.
    """
    streamline_ids = np.asarray(streamline_ids, dtype=np.int32)
    dissimilarity_matrix = dissimilarity_matrix[streamline_ids]

    logging.info(f"Clustering with MKBM with {n_clusters} clusters")
    mbkm = MiniBatchKMeans(
        init="random",
        n_clusters=n_clusters,
        batch_size=1000,
        n_init=10,
        max_no_improvement=5,
        verbose=0,
    )
    mbkm.fit(dissimilarity_matrix)

    medoids_exhs = np.zeros(n_clusters, dtype=np.int32)
    idxs = []
    for i, centroid in enumerate(mbkm.cluster_centers_):
        idx_i = np.where(mbkm.labels_ == i)[0]
        if idx_i.size == 0:
            idx_i = [0]
        tmp = dissimilarity_matrix[idx_i] - centroid
        medoids_exhs[i] = streamline_ids[idx_i[(tmp * tmp).sum(1).argmin()]]
        idxs.append(streamline_ids[idx_i].tolist())

    clusters = dict(zip(medoids_exhs, idxs))
    return clusters
