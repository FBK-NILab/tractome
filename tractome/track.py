import json
import logging
import os
import tempfile

import click
import numpy as np
from dipy.io.streamline import (
    Space,
    StatefulTractogram,
    load_tractogram,
    save_tractogram,
)
from dipy.segment.clustering import qbx_and_merge
from dipy.tracking.distances import bundles_distances_mam
from dipy.tracking.streamline import Streamlines, length
from dipy.utils.optpkg import optional_package

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


def process_tractogram(
    tractogram,
    min_lengths=None,
    min_size=None,
    threshold=15.0,
    num_prototypes=150,
    method="sff",
    output=None,
):
    sft = load_tractogram(tractogram, "same", bbox_valid_check=False)

    streamlines = sft.streamlines

    # print(sft.data_per_streamline.get("dismatrix"))

    if sft.data_per_streamline is not None and "dismatrix" in sft.data_per_streamline:
        data_dissimilarity = sft.data_per_streamline["dismatrix"]
        logging.info(
            "Found dissimilarity matrix in the tractogram of shape: "
            f"({len(data_dissimilarity)}, {len(data_dissimilarity[0])})"
        )
        return
    else:
        if method == "qbx":
            clusters = qbx_and_merge(streamlines, [40, 30, 25, 20, threshold])
            centroids = clusters.centroids
            lengths = [length(c) for c in centroids]
            lengths = np.asarray(lengths)

            logging.info(f"Minimum length of streamlines in cluster {np.min(lengths)}")
            logging.info(f"Maximum length of streamlines in cluster {np.max(lengths)}")

            sizes = [len(c) for c in centroids]
            sizes = np.asarray(sizes)

            logging.info(f"Minimum number of streamlines in cluster {np.min(sizes)}")
            logging.info(f"Maximum number of streamlines in cluster {np.max(sizes)}")

            streamlines = Streamlines()
            for i, c in enumerate(clusters):
                if sizes[i] < min_size:
                    logging.info(f"Removing cluster {i} with size {sizes[i]}")
                    continue
                if lengths[i] < min_lengths:
                    logging.info(f"Removing cluster {i} with length {lengths[i]}")
                    continue
                streamlines.extend(c)
            data_dissimilarity = None
        else:
            data_dissimilarity = compute_dissimilarity(
                np.asarray(streamlines, dtype=object),
                bundles_distances_mam,
                method,
                num_prototypes,
                verbose=True,
            )

    if output is None:
        output = "filtered.trx"
    elif not output.endswith(".trx"):
        output += ".trx"

    logging.info(f"Saving result in {output}")

    sft_new = StatefulTractogram(
        streamlines,
        tractogram,
        Space.RASMM,
        data_per_streamline={"dismatrix": data_dissimilarity},
    )
    save_tractogram(sft_new, output, bbox_valid_check=False)
    logging.info("Saved!")


@click.command()
@click.argument("tractogram", type=click.Path(exists=True))
@click.option(
    "--min_lengths",
    default=100,
    help="Minimum length of streamlines in cluster to keep.",
)
@click.option(
    "--min_size",
    default=10,
    help="Minimum number of streamlines in cluster to keep.",
)
@click.option(
    "--threshold",
    default=15.0,
    help="Threshold for clustering using qbx_and_merge.",
)
@click.option(
    "--num_prototypes",
    default=150,
    help="Number of prototypes to keep.",
)
@click.option(
    "--method",
    default="sff",
    help="Method to use for transformation.",
)
def transform_tract(
    tractogram,
    *,
    min_lengths=100,
    min_size=10,
    threshold=15.0,
    num_prototypes=150,
    method="sff",
):
    """
    Transform a tractogram by removing small clusters and saving the result.
    """
    process_tractogram(
        tractogram, min_lengths, min_size, threshold, num_prototypes, method
    )


if __name__ == "__main__":
    process_tractogram(
        os.path.expanduser(
            "~/.dipy/bundle_atlas_hcp842/Atlas_80_Bundles/whole_brain/whole_brain_MNI.trk"
        ),
        min_lengths=100,
        min_size=10,
        threshold=15.0,
        num_prototypes=150,
        method="sff",
    )
