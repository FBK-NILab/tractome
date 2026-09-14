from concurrent.futures import ThreadPoolExecutor
import json
import logging
import tempfile

import click
from dipy.tracking.distances import bundles_distances_mam, bundles_distances_mdf
from dipy.utils.optpkg import optional_package
import numpy as np
from scipy.ndimage import affine_transform
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

from tractome.io import read_tractogram, save_tractogram

ray, has_ray, _ = optional_package("ray")

# --- Large-tractogram subsampling ---------------------------------------
# Above SUBSAMPLE_THRESHOLD streamlines the interactive working set is reduced
# to a stratified sample of the full tractogram. Strata come from a one-time
# fine MiniBatchKMeans pass (k_s = threshold // q) over the dissimilarity
# embedding; per stratum we always keep the medoid plus a seeded random draw of
# members (at least KEEP_FLOOR). The full-resolution set is recoverable later.
SUBSAMPLE_THRESHOLD = 150_000
FINE_Q = 20  # target kept-per-stratum -> k_s = threshold // q (~7500)
KEEP_FLOOR = 5
SUBSAMPLE_SEED = 0

# --- Fiber recovery ------------------------------------------------------
# Recovery pulls un-shown streamlines back around a selected bundle. The fine
# strata double as a coarse quantizer (an inverted-file index): only the strata
# the bundle occupies, plus their RECOVERY_NPROBE nearest siblings, are searched
# exactly, which keeps the candidate pool at a few thousand rows even on a
# million-streamline tractogram. RECOVERY_QUERY_CAP bounds the bundle side of
# that search so cost stays flat however large the selection grows.
RECOVERY_NPROBE = 8
RECOVERY_QUERY_CAP = 5000
RECOVERY_CHUNK = 20_000


def furthest_first_traversal(S, k, distance, permutation=True):
    """Select prototypes with farthest-first traversal.

    This is a good sub-optimal solution to the k-center problem.

    See for example:
    Hochbaum, Dorit S. and Shmoys, David B., A Best Possible Heuristic
    for the k-Center Problem, Mathematics of Operations Research, 1985.

    or: http://en.wikipedia.org/wiki/Metric_k-center

    Parameters
    ----------
    S : ndarray
        Input samples to select from.
    k : int
        Number of samples to select.
    distance : callable
        Function that computes pairwise distances between samples.
    permutation : bool, optional
        If True, permute ``S`` before selecting prototypes.

    Returns
    -------
    ndarray
        Indices of the selected samples in the original input order.
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
    """Select prototypes from a random subset with farthest-first traversal.

    This is a stochastic scalable version of the FFT algorithm based on a
    random subset of a specific size.

    See: E. Olivetti, T.B. Nguyen, E. Garyfallidis, The Approximation
    of the Dissimilarity Projection, Proceedings of the 2012
    International Workshop on Pattern Recognition in NeuroImaging
    (PRNI), vol., no., pp.85,88, 2-4 July 2012 doi:
    10.1109/PRNI.2012.13

    D. Turnbull and C. Elkan, Fast Recognition of Musical Genres
    Using RBF Networks, IEEE Trans Knowl Data Eng, vol. 2005, no. 4,
    pp. 580-584, 17.

    Parameters
    ----------
    S : ndarray
        Input samples to select from.
    k : int
        Number of samples to select.
    distance : callable
        Function that computes pairwise distances between samples.
    permutation : bool, optional
        If True, sample from a random permutation of ``S``.
    c : float, optional
        Multiplier used to determine the random subset size.

    Returns
    -------
    ndarray
        Indices of the selected samples in the original input order.
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
    """Compute a dissimilarity matrix from selected prototypes.

    Parameters
    ----------
    data : ndarray
        Input streamlines or feature objects.
    distance : callable
        Distance function used to compare ``data`` against prototypes.
    prototype_policy : {'random', 'fft', 'sff'}
        Strategy used to select prototypes.
    num_prototypes : int
        Number of prototypes to select.
    verbose : bool, optional
        If True, emit additional log messages.
    size_limit : int, optional
        Maximum number of samples used for the dissimilarity computation.
    n_jobs : int, optional
        Number of Ray workers to use when Ray is available.

    Returns
    -------
    ndarray or list
        Distances from each sample to the selected prototypes.

    Raises
    ------
    Exception
        If ``prototype_policy`` is unknown.
    """
    logging.info("Computing dissimilarity matrix.")
    data_original = data
    num_proto = num_prototypes
    if data.shape[0] > size_limit:
        logging.info("Dataset too big: subsampling to %s entries only!" % size_limit)
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
                include_dashboard=False,
                _system_config={
                    "object_spilling_config": json.dumps(
                        {
                            "type": "filesystem",
                            "params": {"directory_path": tmp_dir.name},
                        }
                    )
                },
            )

        func = ray.remote(distance)
        func_refs = [func.remote(data[start:end], prototype) for start, end in chunks]

        data_dissimilarity = []
        for i in range(len(func_refs)):
            data_dissimilarity.extend(ray.get(func_refs[i]))

    elif n_jobs > 1:
        logging.info(
            "Threaded computation of the dissimilarity matrix: %s workers." % n_jobs
        )
        tmp = np.linspace(0, data.shape[0], n_jobs).astype(np.int32)
        chunks = [(start, end) for start, end in zip(tmp[:-1], tmp[1:]) if start < end]

        with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
            chunk_results = executor.map(
                lambda chunk: distance(data[chunk[0] : chunk[1]], prototype),
                chunks,
            )

        data_dissimilarity = []
        for chunk_result in chunk_results:
            data_dissimilarity.extend(chunk_result)

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


def fine_cluster_labels(
    dismatrix, n_clusters, *, seed=SUBSAMPLE_SEED, batch_size=10000, n_init=3
):
    """Assign every streamline to a fine MiniBatchKMeans stratum.

    This is the one-time fine pass used to build sampling strata (and, later,
    the recovery index) over the *full* tractogram. Unlike :func:`mkbm_clustering`
    it never loops per-cluster over all rows to find medoids (that is O(k*N) and
    hangs at ``k_s=7500`` on millions of streamlines); medoids are found with a
    single vectorized segmented arg-min.

    Parameters
    ----------
    dismatrix : ndarray
        Dissimilarity embedding of shape ``(N, num_prototypes)``.
    n_clusters : int
        Number of fine strata to create.
    seed : int, optional
        Seed for ``MiniBatchKMeans`` reproducibility.
    batch_size : int, optional
        Mini-batch size (larger than the interactive default for speed).
    n_init : int, optional
        Number of initializations (lower than the interactive default; this is
        a coarse, one-time pass).

    Returns
    -------
    labels : ndarray
        ``int32`` array of shape ``(N,)`` mapping each streamline to a stratum.
    medoid_ids : ndarray
        ``int32`` array of the medoid streamline id of each non-empty stratum
        (the member nearest its centroid). Empty strata are omitted.
    """
    dismatrix = np.asarray(dismatrix, dtype=np.float32)
    n = dismatrix.shape[0]
    n_clusters = max(1, min(int(n_clusters), n))

    mbkm = MiniBatchKMeans(
        init="random",
        n_clusters=n_clusters,
        batch_size=batch_size,
        n_init=n_init,
        max_no_improvement=5,
        random_state=seed,
        verbose=0,
    )
    labels = mbkm.fit_predict(dismatrix).astype(np.int32)

    # Squared distance of every streamline to its own centroid.
    diff = dismatrix - mbkm.cluster_centers_[labels]
    d2 = np.einsum("ij,ij->i", diff, diff)

    # Vectorized per-stratum arg-min: sort by label, then for each contiguous
    # segment pick the row with the smallest d2.
    order = np.argsort(labels, kind="stable")
    sorted_labels = labels[order]
    seg_starts = np.searchsorted(sorted_labels, np.arange(n_clusters), side="left")
    seg_ends = np.searchsorted(sorted_labels, np.arange(n_clusters), side="right")

    medoid_ids = []
    for c in range(n_clusters):
        start, end = seg_starts[c], seg_ends[c]
        if start == end:  # empty stratum (possible with MiniBatchKMeans)
            continue
        seg = order[start:end]
        medoid_ids.append(int(seg[d2[seg].argmin()]))

    return labels, np.asarray(medoid_ids, dtype=np.int32)


def stratified_subsample(
    dismatrix,
    *,
    threshold=SUBSAMPLE_THRESHOLD,
    q=FINE_Q,
    floor=KEEP_FLOOR,
    seed=SUBSAMPLE_SEED,
):
    """Reduce a tractogram to a coverage-preserving stratified sample.

    When the tractogram exceeds ``threshold`` streamlines a fine
    MiniBatchKMeans pass partitions the embedding into ``k_s = threshold // q``
    strata. From each non-empty stratum the medoid is always kept plus a seeded
    random draw of members, so ``keep = min(size, max(floor, round(pct*size)))``
    with ``pct = threshold / N``. This preserves coverage of small structures
    (every stratum contributes at least its medoid) far better than a flat
    uniform sample, while keeping the working set near ``threshold``.

    Parameters
    ----------
    dismatrix : ndarray
        Dissimilarity embedding of shape ``(N, num_prototypes)``.
    threshold : int, optional
        Working-set size ceiling. When ``N <= threshold`` the full tractogram
        is returned unchanged and no fine pass runs.
    q : int, optional
        Target kept-per-stratum, controlling stratum granularity via
        ``k_s = threshold // q``.
    floor : int, optional
        Minimum streamlines kept per non-empty stratum.
    seed : int, optional
        Seed for both the fine pass and the per-stratum random draws.

    Returns
    -------
    dict
        ``{"sample_ids", "labels", "medoid_ids", "k_s", "n"}``. When no
        subsampling is applied, ``labels`` and ``medoid_ids`` are ``None`` and
        ``k_s`` is ``0``.
    """
    dismatrix = np.asarray(dismatrix)
    n = dismatrix.shape[0]

    if n <= threshold:
        return {
            "sample_ids": np.arange(n, dtype=np.int32),
            "labels": None,
            "medoid_ids": None,
            "k_s": 0,
            "n": n,
        }

    k_s = max(1, threshold // q)
    labels, medoid_ids = fine_cluster_labels(dismatrix, k_s, seed=seed)

    duplicate_medoids = len(medoid_ids) - len(np.unique(medoid_ids))
    if duplicate_medoids:
        logging.warning(
            "Fine pass produced %s duplicate medoid ids; strata sharing a "
            "medoid may collapse when re-clustered.",
            duplicate_medoids,
        )

    pct = threshold / n
    rng = np.random.default_rng(seed)

    # Group streamline ids by stratum in one pass.
    order = np.argsort(labels, kind="stable")
    sorted_labels = labels[order]
    n_clusters = int(labels.max()) + 1
    seg_starts = np.searchsorted(sorted_labels, np.arange(n_clusters), side="left")
    seg_ends = np.searchsorted(sorted_labels, np.arange(n_clusters), side="right")
    medoid_set = set(medoid_ids.tolist())

    chosen = []
    for c in range(n_clusters):
        start, end = seg_starts[c], seg_ends[c]
        if start == end:
            continue
        members = order[start:end]
        size = end - start
        keep = min(size, max(floor, int(round(pct * size))))

        # Identify this stratum's medoid (if any) so it is always kept.
        stratum_medoid = None
        for mid in members:
            if int(mid) in medoid_set:
                stratum_medoid = int(mid)
                break

        if stratum_medoid is None:
            chosen.append(rng.choice(members, keep, replace=False))
            continue

        chosen.append(np.asarray([stratum_medoid], dtype=members.dtype))
        remaining = members[members != stratum_medoid]
        extra = keep - 1
        if extra > 0 and len(remaining):
            extra = min(extra, len(remaining))
            chosen.append(rng.choice(remaining, extra, replace=False))

    sample_ids = np.unique(np.concatenate(chosen)).astype(np.int32)
    return {
        "sample_ids": sample_ids,
        "labels": labels,
        "medoid_ids": medoid_ids,
        "k_s": k_s,
        "n": n,
    }


def nearest_reference(reference, queries, *, chunk=RECOVERY_CHUNK):
    """Find, for every query row, its nearest row in ``reference``.

    Brute force is deliberate: the dissimilarity embedding has ~40 dimensions,
    where tree-based indices degenerate to a linear scan anyway while costing a
    build. The brute-force path is BLAS-backed and needs no index at all, and
    the callers here always search a pre-narrowed pool.

    Parameters
    ----------
    reference : ndarray
        Rows to search, of shape ``(R, d)``.
    queries : ndarray
        Rows to look up, of shape ``(Q, d)``.
    chunk : int, optional
        Number of query rows resolved per batch, bounding peak memory of the
        ``(chunk, R)`` distance block.

    Returns
    -------
    distances : ndarray
        ``float32`` distance from each query row to its nearest reference row.
    indices : ndarray
        ``int32`` row index into ``reference`` of that nearest row.
    """
    reference = np.asarray(reference, dtype=np.float32)
    queries = np.asarray(queries, dtype=np.float32)

    distances = np.empty(len(queries), dtype=np.float32)
    indices = np.empty(len(queries), dtype=np.int32)
    if len(queries) == 0 or len(reference) == 0:
        return distances, indices

    nn = NearestNeighbors(n_neighbors=1, algorithm="brute").fit(reference)
    for start in range(0, len(queries), max(1, int(chunk))):
        stop = min(start + max(1, int(chunk)), len(queries))
        batch_distances, batch_indices = nn.kneighbors(queries[start:stop])
        distances[start:stop] = batch_distances[:, 0]
        indices[start:stop] = batch_indices[:, 0]

    return distances, indices


def knn_indices(reference, queries, k):
    """Return the ``k`` nearest reference rows for each query row.

    Parameters
    ----------
    reference : ndarray
        Rows to search, of shape ``(R, d)``.
    queries : ndarray
        Rows to look up, of shape ``(Q, d)``.
    k : int
        Neighbours per query, clamped to ``R``.

    Returns
    -------
    ndarray
        ``int32`` array of shape ``(Q, min(k, R))`` of reference row indices.
    """
    reference = np.asarray(reference, dtype=np.float32)
    queries = np.asarray(queries, dtype=np.float32)
    k = max(1, min(int(k), len(reference)))

    if len(queries) == 0 or len(reference) == 0:
        return np.empty((len(queries), 0), dtype=np.int32)

    nn = NearestNeighbors(n_neighbors=k, algorithm="brute").fit(reference)
    return nn.kneighbors(queries, return_distance=False).astype(np.int32)


def rank_recovery_candidates(
    dismatrix,
    bundle_ids,
    candidate_ids,
    budget,
    *,
    seed=SUBSAMPLE_SEED,
    query_cap=RECOVERY_QUERY_CAP,
):
    """Pick the ``budget`` candidates closest to a bundle in embedding space.

    Each candidate is scored by its distance to the *nearest* member of the
    bundle, so a fiber tight against any part of the bundle ranks highly even
    when the bundle itself is long or curved. Scoring in this direction costs
    one pass over the candidates rather than one per bundle member.

    Parameters
    ----------
    dismatrix : ndarray
        Dissimilarity embedding of the full tractogram, shape ``(N, d)``.
    bundle_ids : array_like
        Streamline ids forming the bundle to grow around.
    candidate_ids : array_like
        Streamline ids eligible to be recovered. Must not overlap
        ``bundle_ids``; callers build this from the un-shown set.
    budget : int
        Maximum number of ids to return.
    seed : int, optional
        Seed used when the bundle is subsampled down to ``query_cap``.
    query_cap : int, optional
        Largest bundle sample used for scoring. Beyond this a seeded random
        subset stands in for the bundle, which bounds cost without meaningfully
        moving the ranking (neighbours of a dense bundle are neighbours of
        almost any dense sample of it).

    Returns
    -------
    ids : ndarray
        ``int32`` recovered streamline ids, closest first.
    distances : ndarray
        ``float32`` distance of each returned id to the bundle, ascending.
    """
    dismatrix = np.asarray(dismatrix, dtype=np.float32)
    bundle_ids = np.asarray(bundle_ids, dtype=np.int32).reshape(-1)
    candidate_ids = np.asarray(candidate_ids, dtype=np.int32).reshape(-1)
    budget = int(budget)

    empty = (np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32))
    if budget <= 0 or bundle_ids.size == 0 or candidate_ids.size == 0:
        return empty

    query_ids = bundle_ids
    if query_cap and query_ids.size > query_cap:
        rng = np.random.default_rng(seed)
        query_ids = rng.choice(query_ids, int(query_cap), replace=False)

    distances, _ = nearest_reference(dismatrix[query_ids], dismatrix[candidate_ids])

    keep = min(budget, candidate_ids.size)
    if keep < candidate_ids.size:
        # argpartition finds the k smallest without ordering the rest.
        selected = np.argpartition(distances, keep - 1)[:keep]
    else:
        selected = np.arange(candidate_ids.size)

    selected = selected[np.argsort(distances[selected], kind="stable")]
    return candidate_ids[selected], distances[selected]


def assign_to_nearest_medoid(dismatrix, streamline_ids, medoid_ids):
    """Attach each streamline to the cluster whose medoid it is closest to.

    Mirrors the assignment step of the original Tractome: recovered fibers join
    an existing cluster instead of forcing a re-clustering, so cluster colors
    and identities survive the operation.

    Parameters
    ----------
    dismatrix : ndarray
        Dissimilarity embedding of the full tractogram, shape ``(N, d)``.
    streamline_ids : array_like
        Streamline ids to assign.
    medoid_ids : array_like
        Streamline ids of the candidate cluster medoids. In tractome a cluster
        is keyed by its own medoid's streamline id, so these double as the
        cluster ids returned.

    Returns
    -------
    ndarray
        ``int32`` cluster id chosen for each entry of ``streamline_ids``.
    """
    dismatrix = np.asarray(dismatrix, dtype=np.float32)
    streamline_ids = np.asarray(streamline_ids, dtype=np.int32).reshape(-1)
    medoid_ids = np.asarray(medoid_ids, dtype=np.int32).reshape(-1)

    if streamline_ids.size == 0 or medoid_ids.size == 0:
        return np.empty(0, dtype=np.int32)

    _, nearest = nearest_reference(dismatrix[medoid_ids], dismatrix[streamline_ids])
    return medoid_ids[nearest].astype(np.int32)


def calculate_filter(rois, *, flip=None, reference_shape=None):
    """Calculate a combined ROI filter using logical AND.

    Parameters
    ----------
    rois : sequence of ndarray
        ROI volumes to combine with shape ``(X, Y, Z)``.
    flip : Sequence[bool] or None, optional
        Per-ROI flag indicating whether the ROI should be inverted
        before combination. If None, no ROIs are inverted.
    reference_shape : tuple[int, ...] or None, optional
        Expected ROI shape. If None, shape from the first ROI is used.

    Returns
    -------
    ndarray
        Boolean mask resulting from a logical AND across all ROIs
        (after optional inversion).

    Raises
    ------
    ValueError
        If no ROIs are provided, if `flip` length does not match
        `rois`, or if no ROI matches `reference_shape`.
    """
    if rois is None or len(rois) == 0:
        raise ValueError("At least one ROI must be provided.")

    if flip is None:
        flip = [False] * len(rois)

    if len(flip) != len(rois):
        raise ValueError(
            "The `flip` list must have the same length as `rois` "
            f"({len(flip)} != {len(rois)})."
        )

    if reference_shape is None:
        reference_shape = np.asarray(rois[0]).shape[:3]
    else:
        reference_shape = tuple(reference_shape)[:3]
    combined_mask = np.ones(reference_shape, dtype=bool)
    matched_count = 0

    for idx, (roi, should_flip) in enumerate(zip(rois, flip)):
        roi_mask = np.asarray(roi).astype(bool, copy=False)

        if roi_mask.shape[:3] != reference_shape:
            logging.warning(
                "Skipping ROI %s due to shape mismatch: expected %s, got %s.",
                idx,
                reference_shape,
                roi_mask.shape,
            )
            continue
        if roi_mask.ndim > 3:
            roi_mask = np.any(roi_mask, axis=tuple(range(3, roi_mask.ndim)))

        if bool(should_flip):
            roi_mask = np.logical_not(roi_mask)

        combined_mask = np.logical_and(combined_mask, roi_mask)
        matched_count += 1

    if matched_count == 0:
        raise ValueError(
            f"No ROI matched the reference shape. Expected shape: {reference_shape}."
        )

    return combined_mask


def create_roi_from_world(bounds, affine, center, radius, *, type="spherical"):
    """Create a binary spherical ROI from world-space center and radius.

    Parameters
    ----------
    bounds : tuple[int, int, int]
        ROI output shape in voxel coordinates.
    affine : ndarray, shape (4, 4)
        Voxel-to-world affine transform.
    center : Sequence[float]
        Sphere center in world coordinates.
    radius : float
        Sphere radius in world units.
    type : str, optional
        Type of ROI to create. Currently only ``"spherical"`` is supported.

    Returns
    -------
    tuple[ndarray, ndarray]
        `(roi, affine)` where `roi` is a uint8 binary mask with ones inside
        the sphere and zeros elsewhere.

    Raises
    ------
    ValueError
        If ``bounds``, ``affine``, ``center``, or ``radius`` is invalid.
    """
    bounds = tuple(int(v) for v in bounds)
    if len(bounds) != 3:
        raise ValueError(f"`bounds` must have 3 dimensions, got {bounds}.")

    affine = np.asarray(affine, dtype=np.float64)
    if affine.shape != (4, 4):
        raise ValueError(f"`affine` must have shape (4, 4), got {affine.shape}.")

    center = np.asarray(center, dtype=np.float64)
    if center.shape != (3,):
        raise ValueError(f"`center` must have shape (3,), got {center.shape}.")
    if radius < 0:
        raise ValueError("`radius` must be non-negative.")

    roi = np.zeros(bounds, dtype=np.uint8)

    inv_affine = np.linalg.inv(affine)
    center_vox = (inv_affine @ np.r_[center, 1.0])[:3]
    inv_linear = inv_affine[:3, :3]
    voxel_radii = np.linalg.norm(inv_linear * float(radius), axis=0)
    radius_vox = float(np.mean(voxel_radii))

    grid = np.indices(bounds, dtype=np.float32)
    dist_sq = (
        (grid[0] - center_vox[0]) ** 2
        + (grid[1] - center_vox[1]) ** 2
        + (grid[2] - center_vox[2]) ** 2
    )
    roi[dist_sq <= (radius_vox**2)] = 1

    return roi, affine


def transform_roi_to_world_grid(roi_data, affine, *, cval=0.0, threshold=0.5):
    """Resample ROI data to an axis-aligned world-coordinate grid.

    The returned array is indexed in world space using `world_min` as origin:
    `world_index = world_coord - world_min`.

    Parameters
    ----------
    roi_data : ndarray
        Input ROI volume in voxel coordinates.
    affine : ndarray, shape (4, 4)
        Voxel-to-world affine transform.
    cval : float, optional
        Constant value for out-of-bounds sampling.
    threshold : float or None, optional
        If not None, output is binarized with ``>= threshold``.

    Returns
    -------
    tuple[ndarray, ndarray]
        `(transformed_data, world_min)` where:
        - `transformed_data` is the ROI in world-grid indexing.
        - `world_min` is the minimum world coordinate (x, y, z) used as origin.

    Raises
    ------
    ValueError
        If `roi_data` is not 3D or `affine` is not 4x4.
    """
    roi_data = np.asarray(roi_data)
    if roi_data.ndim != 3:
        raise ValueError(f"`roi_data` must be a 3D array, got shape {roi_data.shape}.")

    affine = np.asarray(affine, dtype=np.float64)
    if affine.shape != (4, 4):
        raise ValueError(f"`affine` must have shape (4, 4), got {affine.shape}.")

    if np.linalg.det(affine[:3, :3]) == 0:
        raise ValueError("`affine` is singular and cannot be inverted.")

    max_idx = np.asarray(roi_data.shape, dtype=np.float64) - 1.0
    corners_ijk = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [max_idx[0], 0.0, 0.0],
            [0.0, max_idx[1], 0.0],
            [0.0, 0.0, max_idx[2]],
            [max_idx[0], max_idx[1], 0.0],
            [max_idx[0], 0.0, max_idx[2]],
            [0.0, max_idx[1], max_idx[2]],
            [max_idx[0], max_idx[1], max_idx[2]],
        ],
        dtype=np.float64,
    )
    corners_h = np.c_[corners_ijk, np.ones(len(corners_ijk), dtype=np.float64)]
    world_corners = (affine @ corners_h.T).T[:, :3]

    world_min = np.floor(world_corners.min(axis=0)).astype(np.int32)
    world_max = np.ceil(world_corners.max(axis=0)).astype(np.int32)
    output_shape = tuple((world_max - world_min + 1).astype(np.int32))

    inv_affine = np.linalg.inv(affine)
    matrix = inv_affine[:3, :3]
    offset = (inv_affine[:3, :3] @ world_min) + inv_affine[:3, 3]

    transformed_data = affine_transform(
        roi_data.astype(np.float32, copy=False),
        matrix=matrix,
        offset=offset,
        output_shape=output_shape,
        order=1,
        mode="constant",
        cval=float(cval),
        prefilter=False,
    )

    if threshold is not None:
        transformed_data = transformed_data >= threshold

    return transformed_data, world_min


def _fetch_positions_from_gpu(show_manager, geom_positions_buffer, *, sync_cpu=False):
    """Read back geometry.positions from GPU into a NumPy array.

    Parameters
    ----------
    show_manager : fury.window.ShowManager
        Show manager whose device owns the GPU buffer.
    geom_positions_buffer : object
        Geometry positions buffer with an attached ``_wgpu_object``.
    sync_cpu : bool, optional
        If True, copy the read-back positions into the CPU-side buffer data.

    Returns
    -------
    ndarray or None
        GPU positions as a NumPy array, or None if no GPU buffer exists.

    Notes
    -----
    This uses pygfx/wgpu internals (`_wgpu_object`) and requires COPY_SRC usage.
    """
    wgpu_buffer = getattr(geom_positions_buffer, "_wgpu_object", None)
    if wgpu_buffer is None:
        return None

    raw = show_manager.device.queue.read_buffer(wgpu_buffer)
    cpu_shape = np.asarray(geom_positions_buffer.data).shape
    gpu_positions = np.frombuffer(raw, dtype=np.float32).reshape(cpu_shape).copy()

    if sync_cpu and geom_positions_buffer.data is not None:
        np.asarray(geom_positions_buffer.data)[...] = gpu_positions

    return gpu_positions


def _get_line_ids_from_positions(wobj, positions):
    """Return kept and filtered line ids from a positions buffer.

    Parameters
    ----------
    wobj : object
        Streamlines actor with ``_line_offsets`` and ``_line_lengths`` metadata.
    positions : ndarray
        Flat position buffer read back from the actor.

    Returns
    -------
    kept_ids : list[int]
        Line ids whose positions are finite.
    filtered_ids : list[int]
        Line ids whose positions contain non-finite values.
    """
    positions = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
    offsets = np.asarray(wobj._line_offsets, dtype=np.int64)
    lengths = np.asarray(wobj._line_lengths, dtype=np.int64)

    kept_ids = []
    filtered_ids = []
    for line_id, (offset, length) in enumerate(zip(offsets, lengths)):
        segment = positions[offset : offset + length]
        if np.isfinite(segment).all():
            kept_ids.append(line_id)
        else:
            filtered_ids.append(line_id)

    return kept_ids, filtered_ids


def filter_streamline_ids(streamlines, roi, *, origin=(0, 0, 0)):
    """Return streamlines that pass through an ROI mask.

    Parameters
    ----------
    streamlines : sequence of ndarray
        Streamlines to filter.
    roi : ndarray
        Binary ROI mask in world-grid coordinates.
    origin : tuple of int, optional
        Origin of ``roi`` in world-grid coordinates.

    Returns
    -------
    list[int]
        Indices of streamlines that intersect the ROI mask.
    """
    # Imported lazily: fury pulls in Qt at import time, so keeping this out of
    # the module scope lets the dissimilarity/embedding path stay Qt-free.
    import wgpu

    from fury import actor, window

    max = np.asarray(streamlines[0], dtype=np.float32).max(axis=0)
    min = np.asarray(streamlines[0], dtype=np.float32).min(axis=0)

    scene = window.Scene()
    filtered_streamlines = actor.streamlines(
        streamlines, roi_mask=roi, roi_origin=origin
    )
    points = actor.point(
        np.asarray([min, max], dtype=np.float32),
        colors=np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
    )
    filtered_streamlines.geometry.positions._wgpu_usage |= wgpu.BufferUsage.COPY_SRC
    offscreen_showm = window.ShowManager(scene=scene, window_type="offscreen")
    scene.add(points)
    scene.add(filtered_streamlines)
    offscreen_showm.render()
    offscreen_showm.window.draw()
    filtered_positions = _fetch_positions_from_gpu(
        offscreen_showm, filtered_streamlines.geometry.positions
    )
    filtered_positions = np.asarray(filtered_positions, dtype=np.float32).reshape(-1, 3)
    kept_ids, _ = _get_line_ids_from_positions(filtered_streamlines, filtered_positions)
    offscreen_showm.close()
    return kept_ids


@click.command(name="compute_dissimilarity_matrix")
@click.argument(
    "tractogram_path",
    type=click.Path(exists=True),
)
@click.option(
    "--reference",
    type=click.Path(exists=True),
    help="Path to the reference image file.",
)
@click.option(
    "--distance",
    type=click.Choice(["bundles_distances_mam", "bundles_distances_mdf"]),
    default="bundles_distances_mam",
    help=(
        "Distance metric to use. Must be one of ['bundles_distances_mam',"
        "'bundles_distances_mdf']."
    ),
)
@click.option(
    "--prototype_policy",
    type=click.Choice(["random", "fft", "sff"]),
    default="sff",
    help="Prototype selection policy. Must be one of ['random', 'fft', 'sff'].",
)
@click.option(
    "--num_prototypes",
    type=int,
    default=40,
    help="Number of prototypes to generate.",
)
@click.option("--verbose", is_flag=True, help="Enable verbose output.")
@click.option(
    "--size_limit",
    type=int,
    default=5_000_000,
    help="Maximum size of the dataset to process.",
)
@click.option("--n_jobs", type=int, default=6, help="Number of parallel jobs.")
@click.option(
    "--output_file",
    type=click.Path(),
    default="computed.trx",
    help=(
        "File path to save the computed dissimilarity matrix along with the tractogram."
    ),
)
def compute_dissimilarity_matrix(
    tractogram_path,
    reference=None,
    distance="bundles_distances_mam",
    prototype_policy="sff",
    num_prototypes=40,
    verbose=False,
    size_limit=5_000_000,
    n_jobs=6,
    output_file="computed.trx",
):
    """Compute the dissimilarity matrix for the given tractogram.

    Parameters
    ----------
    tractogram_path : str
        The path to the tractogram file.
    reference : str, optional
        The path to the reference image file.
    distance : str, optional
        The distance metric to use.
    prototype_policy : str, optional
        The prototype selection policy to use.
        Must be one of ['random', 'fft', 'sff'].
    num_prototypes : int, optional
        The number of prototypes to generate.
    verbose : bool, optional
        If True, enables verbose output.
    size_limit : int, optional
        Maximum size of the dataset to process.
    n_jobs : int, optional
        The number of parallel jobs to run.
    output_file : str, optional
        The file path to save the computed dissimilarity matrix along with the
        tractogram.

    Raises
    ------
    ValueError
        The distance metric to use.
    ValueError
        The prototype selection policy to use.
    """
    sft = read_tractogram(tractogram_path, reference=reference)

    if distance == "bundles_distances_mam":
        distance = bundles_distances_mam
    elif distance == "bundles_distances_mdf":
        distance = bundles_distances_mdf
    else:
        raise ValueError(
            f"Unknown distance metric: {distance}, must be one of "
            "['bundles_distances_mam', 'bundles_distances_mdf']"
        )

    if prototype_policy not in ["random", "fft", "sff"]:
        raise ValueError(
            f"Unknown prototype policy: {prototype_policy},"
            "must be one of ['random', 'fft', 'sff']"
        )

    data_dissimilarity = compute_dissimilarity(
        np.asarray(sft.streamlines, dtype=object),
        distance=distance,
        prototype_policy=prototype_policy,
        num_prototypes=num_prototypes,
        verbose=verbose,
        size_limit=size_limit,
        n_jobs=n_jobs,
    )
    sft.data_per_streamline["dismatrix"] = data_dissimilarity
    save_tractogram(sft, output_file)
