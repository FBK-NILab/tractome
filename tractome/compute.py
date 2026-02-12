import json
import logging
import tempfile

from dipy.utils.optpkg import optional_package
import numpy as np
from scipy.ndimage import affine_transform
from sklearn.cluster import MiniBatchKMeans
import wgpu

from fury import actor, window

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


def calculate_filter(rois, *, flip=None, reference_shape=None):
    """Calculate a combined ROI filter using logical AND.

    Parameters
    ----------
    rois : ndarray
        ROI volumes to combine with shape (X, Y, Z).
    flip : Sequence[bool] or None, optional
        Per-ROI flag indicating whether the ROI should be inverted
        before combination. If None, all ROIs are inverted.
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
        flip = [True] * len(rois)

    if len(flip) != len(rois):
        raise ValueError(
            "The `flip` list must have the same length as `rois` "
            f"({len(flip)} != {len(rois)})."
        )

    if reference_shape is None:
        reference_shape = np.asarray(rois[0]).shape
    else:
        reference_shape = tuple(reference_shape)
    combined_mask = np.ones(reference_shape, dtype=bool)
    matched_count = 0

    for idx, (roi, should_flip) in enumerate(zip(rois, flip)):
        roi_mask = np.asarray(roi).astype(bool, copy=False)

        if roi_mask.shape != reference_shape:
            logging.warning(
                "Skipping ROI %s due to shape mismatch: expected %s, got %s.",
                idx,
                reference_shape,
                roi_mask.shape,
            )
            continue

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
        Type of ROI to create. Currently only "spherical" is supported.

    Returns
    -------
    tuple[ndarray, ndarray]
        `(roi, affine)` where `roi` is a uint8 binary mask with ones inside
        the sphere and zeros elsewhere.
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
    """Return kept/filtered original line ids from a flat positions buffer."""
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
