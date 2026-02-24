import click
from dipy.tracking.distances import bundles_distances_mam, bundles_distances_mdf
import numpy as np

from tractome.app import Tractome
from tractome.compute import compute_dissimilarity
from tractome.io import read_tractogram, save_tractogram


@click.command(name="tractome")
@click.option(
    "--tractogram", type=click.Path(exists=True), help="Path to the tractogram file."
)
@click.option("--mesh", type=click.Path(exists=True), help="Path to the mesh file.")
@click.option(
    "--mesh_texture",
    type=click.Path(exists=True),
    help="Path to the mesh texture file.",
)
@click.option(
    "--t1", type=click.Path(exists=True), help="Path to the T1-weighted image file."
)
@click.option(
    "--roi",
    type=click.Path(exists=True),
    multiple=True,
    help="Path to an ROI file. Use multiple times for multiple ROIs.",
)
@click.option(
    "--parcel",
    type=click.Path(exists=True),
    multiple=True,
    help=(
        "Path to a parcel CSV file or directory containing CSV files. "
        "Use multiple times for multiple parcels."
    ),
)
def tractome(tractogram=None, mesh=None, mesh_texture=None, t1=None, roi=(), parcel=()):
    """Run the Tractome pipeline.

    Parameters
    ----------
    tractogram : str, optional
        Path to the tractogram file
    mesh : str, optional
        Path to the mesh file
    mesh_texture : str, optional
        Path to the mesh texture file
    t1 : str, optional
        Path to the T1-weighted image file
    roi : tuple[str], optional
        One or more paths to ROI files
    parcel : tuple[str], optional
        One or more paths to parcel files
    """
    Tractome(tractogram, mesh, mesh_texture, t1, roi, parcel)


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
