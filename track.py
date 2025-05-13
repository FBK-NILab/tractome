import logging
import os

import click
import numpy as np
from dipy.io.streamline import (
    Space,
    StatefulTractogram,
    load_tractogram,
    save_tractogram,
)
from dipy.segment.clustering import qbx_and_merge
from dipy.tracking.streamline import Streamlines, length


def process_tractogram(
    tractogram, min_lengths=None, min_size=None, threshold=15.0, output=None
):
    sft = load_tractogram(tractogram, "same", bbox_valid_check=False)

    streamlines = sft.streamlines

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

    saving_streamlines = Streamlines()
    for i, c in enumerate(clusters):
        if sizes[i] < min_size:
            logging.info(f"Removing cluster {i} with size {sizes[i]}")
            continue
        if lengths[i] < min_lengths:
            logging.info(f"Removing cluster {i} with length {lengths[i]}")
            continue
        saving_streamlines.extend(c)

    if output is None:
        output = "filtered.trk"
    elif not output.endswith(".trk"):
        output += ".trk"

    logging.info(f"Saving result in {output}")

    sft_new = StatefulTractogram(saving_streamlines, tractogram, Space.RASMM)
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
    help="Threshold for clustering.",
)
def transform_tract(tractogram, min_lengths=100, min_size=10, threshold=15.0):
    """
    Transform a tractogram by removing small clusters and saving the result.
    """
    process_tractogram(tractogram, min_lengths, min_size, threshold)


if __name__ == "__main__":
    process_tractogram(
        os.path.expanduser(
            "~/.dipy/bundle_atlas_hcp842/Atlas_80_Bundles/whole_brain/whole_brain_MNI.trk"
        ),
        min_lengths=100,
        min_size=10,
        threshold=15.0,
    )
