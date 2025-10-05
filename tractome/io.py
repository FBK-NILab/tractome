import logging
import os

from dipy.io.image import load_nifti
from dipy.io.streamline import load_tractogram, save_tractogram as dipy_save_tractogram
import trimesh


def validate_path(path):
    """Validate the provided file path.

    Parameters
    ----------
    path : str
        The file path to validate.

    Returns
    -------
    str
        The expanded user path if valid.

    Raises
    ------
    FileNotFoundError
        If the file does not exist or is not a file.
    """
    path = os.path.expanduser(path)
    if os.path.exists(path) and os.path.isfile(path):
        return path
    else:
        raise FileNotFoundError(f"The file {path} does not exist or is not a file.")


def read_tractogram(file_path, reference=None):
    """Read a tractogram file.

    Parameters
    ----------
    file_path : str
        The path to the tractogram file.

    Returns
    -------
    tractogram : StatefulTractogram
        The loaded tractogram.
    """

    validated_path = validate_path(file_path)
    logging.info(f"Loading tractogram from {validated_path} ...")

    if reference is None:
        if validated_path.endswith((".trk", ".trx")):
            reference = "same"
        else:
            raise ValueError(
                "Reference image must be provided for files other than "
                ".trk and .trx files."
            )

    sft = load_tractogram(validated_path, reference, bbox_valid_check=False)
    if not sft:
        raise ValueError(f"Failed to load tractogram from {validated_path}.")

    if sft.data_per_streamline is not None and "dismatrix" in sft.data_per_streamline:
        logging.info("Dissimilarity matrix already present in the tractogram data.")

    return sft


def read_mesh(file_path, *, texture=None):
    """Read a mesh file.

    Parameters
    ----------
    file_path : str
        The path to the mesh file.
    texture : str, optional
        The path to a texture file, if applicable.

    Returns
    -------
    tuple: (mesh, texture)
        The loaded mesh and texture (if provided).
    """
    validated_path = validate_path(file_path)
    logging.info(f"Loading mesh from {validated_path} ...")

    mesh = trimesh.load_mesh(validated_path)

    if texture:
        texture = validate_path(texture)
        logging.info(f"Validating texture from {texture} ...")

    return mesh, texture


def read_nifti(file_path):
    """Read a NIfTI file.

    Parameters
    ----------
    file_path : str
        The path to the NIfTI file.

    Returns
    -------
    tuple: (nifti_img, affine)
        The loaded NIfTI image and its affine transformation matrix.
    """

    validated_path = validate_path(file_path)
    logging.info(f"Loading NIfTI file from {validated_path} ...")

    nifti_img, affine = load_nifti(validated_path)

    return nifti_img, affine


def save_tractogram(sft, file_path):
    """Save a tractogram to a file.

    Parameters
    ----------
    sft : StatefulTractogram
        The tractogram to save.
    file_path : str
        The path where the tractogram will be saved.
    """

    validated_path = os.path.expanduser(file_path)
    logging.info(f"Saving tractogram to {validated_path} ...")

    dipy_save_tractogram(sft, validated_path, bbox_valid_check=False)
    logging.info("Tractogram saved successfully.")
