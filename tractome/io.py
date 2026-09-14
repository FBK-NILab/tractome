import csv
from dataclasses import dataclass
import logging
import os

from dipy.io.image import load_nifti, save_nifti
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import load_tractogram, save_tractogram as dipy_save_tractogram
import numpy as np
import polyxios as px


def get_file_extension(file_path):
    """Get the file extension from a file path.

    Parameters
    ----------
    file_path : str
        The file path to extract the extension from.

    Returns
    -------
    str
        The file extension, including the leading dot (e.g., '.trk').
    """
    _, ext = os.path.splitext(file_path)
    return ext.lower()


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


EMBEDDING_LABELS = {"dismatrix": "dissimilarity"}


def get_embedding_label(embedding_name):
    """Return the user-facing label for a stored embedding key.

    Parameters
    ----------
    embedding_name : str
        The ``data_per_streamline`` key as stored in the tractogram.

    Returns
    -------
    str
        The label to display for the embedding (falls back to the key
        itself when no mapping is defined).
    """
    return EMBEDDING_LABELS.get(embedding_name, embedding_name)


def get_embedding_keys(sft):
    """List the per-streamline embeddings stored in a tractogram.

    An embedding is any ``data_per_streamline`` entry whose per-streamline
    value is a vector (width >= 2). The entry's key doubles as the human
    readable embedding name/type (e.g. ``"dissimilarity"``, ``"finta"``).
    No naming convention is assumed. Entries whose row count does not match
    the number of streamlines are treated as corrupted and skipped.

    Parameters
    ----------
    sft : StatefulTractogram
        The tractogram to inspect.

    Returns
    -------
    list[str]
        Names of the available embeddings, in insertion order.
    """
    data_per_streamline = getattr(sft, "data_per_streamline", None)
    if not data_per_streamline:
        return []

    n_streamlines = len(sft.streamlines)
    keys = []
    for key in data_per_streamline.keys():
        try:
            values = np.asarray(data_per_streamline[key])
        except (ValueError, TypeError):
            logging.warning(f"Skipping unreadable data_per_streamline entry '{key}'.")
            continue
        if values.ndim != 2 or values.shape[1] < 2:
            continue
        if values.shape[0] != n_streamlines:
            logging.warning(
                f"Skipping corrupted embedding '{key}': "
                f"{values.shape[0]} rows for {n_streamlines} streamlines."
            )
            continue
        keys.append(key)
    return keys


def read_tractogram(file_path, reference=None):
    """Read a tractogram file.

    Parameters
    ----------
    file_path : str
        The path to the tractogram file.
    reference : str or Nifti1Image, optional
        The reference image for the tractogram.

    Returns
    -------
    StatefulTractogram
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

    embedding_keys = get_embedding_keys(sft)
    if embedding_keys:
        logging.info(f"Embeddings found in the tractogram data: {embedding_keys}.")
    else:
        logging.info("No embeddings found in the tractogram data.")

    return sft


@dataclass(frozen=True)
class MeshData:
    """Lightweight mesh container returned by :func:`read_mesh`."""

    vertices: np.ndarray
    faces: np.ndarray | None
    normals: np.ndarray | None = None
    texcoords: np.ndarray | None = None


# Known per-format UV attribute name pairs in polyxios vertex_attrs.
_UV_ATTR_PAIRS = [("s", "t"), ("texture_u", "texture_v")]


def _extract_texcoords_from_attrs(attrs):
    """Return (N, 2) float32 UVs from vertex_attrs, or None."""
    for u_key, v_key in _UV_ATTR_PAIRS:
        if u_key in attrs and v_key in attrs:
            u = np.asarray(attrs[u_key], dtype=np.float32)
            v = np.asarray(attrs[v_key], dtype=np.float32)
            return np.column_stack([u, v])
    return None


def _read_obj_with_texcoords(path):
    """Re-parse an OBJ to build a mesh with vertices split at UV seams.

    OBJ allows separate indices for positions (v), texture coordinates
    (vt) and normals (vn) per face corner.  A single position that
    touches different UVs at a seam must become multiple vertices so the
    GPU gets a 1-to-1 vertex-to-UV mapping.  This function builds that
    expanded mesh and returns the four arrays that :class:`MeshData`
    expects.
    """
    positions: list[list[float]] = []
    texcoords: list[list[float]] = []
    normals: list[list[float]] = []
    faces_raw: list[list[tuple[int, int | None, int | None]]] = []

    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            directive = parts[0].lower()

            if directive == "v":
                positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif directive == "vt":
                texcoords.append(
                    [float(parts[1]), float(parts[2]) if len(parts) > 2 else 0.0]
                )
            elif directive == "vn":
                normals.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif directive == "f":
                face: list[tuple[int, int | None, int | None]] = []
                for tok in parts[1:]:
                    c = tok.split("/")
                    vi = int(c[0]) - 1
                    vti = (int(c[1]) - 1) if len(c) >= 2 and c[1] else None
                    vni = (int(c[2]) - 1) if len(c) >= 3 and c[2] else None
                    face.append((vi, vti, vni))
                faces_raw.append(face)

    if not texcoords:
        return None, None, None, None

    unique: dict[tuple[int, int | None, int | None], int] = {}
    new_pos: list[list[float]] = []
    new_uv: list[list[float]] = []
    new_nrm: list[list[float]] = []
    tri_faces: list[list[int]] = []

    for face in faces_raw:
        idx: list[int] = []
        for key in face:
            if key not in unique:
                vi, vti, vni = key
                unique[key] = len(new_pos)
                new_pos.append(positions[vi])
                new_uv.append(texcoords[vti] if vti is not None else [0.0, 0.0])
                if normals and vni is not None:
                    new_nrm.append(normals[vni])
            idx.append(unique[key])
        for i in range(1, len(idx) - 1):
            tri_faces.append([idx[0], idx[i], idx[i + 1]])

    verts = np.array(new_pos, dtype=np.float32)
    faces_arr = np.array(tri_faces, dtype=np.int32)
    uvs = np.array(new_uv, dtype=np.float32)
    nrm_arr = np.array(new_nrm, dtype=np.float32) if new_nrm else None
    return verts, faces_arr, nrm_arr, uvs


def read_mesh(file_path, *, texture=None):
    """Read a mesh file using polyxios.

    Parameters
    ----------
    file_path : str
        The path to the mesh file.
    texture : str, optional
        The path to a texture file, if applicable.

    Returns
    -------
    mesh : MeshData
        The loaded mesh data.
    texture : str or None
        Validated texture path, or None if no texture was provided.
    """
    validated_path = validate_path(file_path)
    logging.info(f"Loading mesh from {validated_path} ...")

    poly = px.read(validated_path)
    vertices = np.asarray(poly.vertices, dtype=np.float32)
    faces = poly.faces
    if faces is not None:
        faces = np.asarray(faces, dtype=np.int32)

    normals = poly.vertex_attrs.get("normals")
    if normals is not None:
        normals = np.asarray(normals, dtype=np.float32)

    texcoords = None
    if texture:
        texcoords = _extract_texcoords_from_attrs(poly.vertex_attrs)
        if texcoords is None and validated_path.lower().endswith(".obj"):
            obj_verts, obj_faces, obj_nrm, obj_uvs = _read_obj_with_texcoords(
                validated_path
            )
            if obj_uvs is not None:
                vertices, faces, texcoords = obj_verts, obj_faces, obj_uvs
                if obj_nrm is not None:
                    normals = obj_nrm

    mesh = MeshData(
        vertices=vertices, faces=faces, normals=normals, texcoords=texcoords
    )

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
    nifti_img : ndarray
        The loaded NIfTI image data.
    affine : ndarray
        The affine transformation matrix.
    """

    validated_path = validate_path(file_path)
    logging.info(f"Loading NIfTI file from {validated_path} ...")

    nifti_img, affine = load_nifti(validated_path)

    return nifti_img, affine


def read_csv(file_path, *, delimiter=",", has_header=True, encoding="utf-8"):
    """Read a CSV file.

    Parameters
    ----------
    file_path : str
        The path to the CSV file.
    delimiter : str, optional
        The CSV delimiter character.
    has_header : bool, optional
        Whether the CSV file contains a header row.
    encoding : str, optional
        The file encoding.

    Returns
    -------
    points : ndarray
        First three columns from all loaded CSV rows.
    colors : ndarray
        Remaining columns from all loaded CSV rows.

    Raises
    ------
    ValueError
        If ``file_path`` is a directory with no CSV files, or a non-CSV file.
    """

    resolved_path = os.path.expanduser(file_path)
    csv_paths = []
    if os.path.isdir(resolved_path):
        csv_paths = sorted(
            os.path.join(resolved_path, name)
            for name in os.listdir(resolved_path)
            if os.path.isfile(os.path.join(resolved_path, name))
            and name.lower().endswith(".csv")
        )
        if not csv_paths:
            raise ValueError(f"No CSV files found in directory: {resolved_path}")
        logging.info(f"Loading CSV files from directory {resolved_path} ...")
    else:
        validated_path = validate_path(resolved_path)
        if not validated_path.lower().endswith(".csv"):
            raise ValueError(f"File must be a CSV: {validated_path}")
        csv_paths = [validated_path]
        logging.info(f"Loading CSV file from {validated_path} ...")

    data_chunks = []
    for csv_path in csv_paths:
        with open(csv_path, newline="", encoding=encoding) as csv_file:
            if has_header:
                rows = list(csv.DictReader(csv_file, delimiter=delimiter))
                chunk = np.asarray([[row[key] for key in row] for row in rows])
            else:
                chunk = np.asarray(list(csv.reader(csv_file, delimiter=delimiter)))
            if chunk.size == 0:
                continue
            data_chunks.append(chunk)

    if not data_chunks:
        return np.empty((0, 3)), np.empty((0, 0))

    data = np.concatenate(data_chunks, axis=0)
    return data[:, :3], data[:, 3:]


def save_tractogram_from_streamlines(
    streamlines,
    reference,
    embeddings,
    *,
    embedding_name="dismatrix",
    file_path="saved.trx",
):
    """Save a tractogram from streamlines to a file.

    Parameters
    ----------
    streamlines : list or ndarray
        The streamlines to save.
    reference : str or Nifti1Image
        The reference image for the tractogram.
    embeddings : ndarray
        The embeddings to attach to the tractogram.
    embedding_name : str, optional
        The name/type under which the embeddings are stored. The name
        doubles as the label shown when selecting an embedding at load time.
    file_path : str, optional
        The path where the tractogram will be saved.
    """

    sft = StatefulTractogram(
        streamlines,
        reference,
        Space.RASMM,
        data_per_streamline={embedding_name: embeddings},
    )
    dipy_save_tractogram(sft, file_path, bbox_valid_check=False)
    logging.info("Tractogram saved successfully.")


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


def save_roi(fpath, roi, affine):
    """Save an ROI as a NIfTI file with uint8 dtype.

    Parameters
    ----------
    fpath : str
        Destination path.
    roi : ndarray
        ROI volume data.
    affine : ndarray
        Voxel-to-world affine matrix.
    """
    validated_path = os.path.expanduser(fpath)
    logging.info(f"Saving ROI to {validated_path} ...")

    roi_uint8 = np.asarray(roi, dtype=np.uint8)
    save_nifti(validated_path, roi_uint8, affine, dtype=np.uint8)
    logging.info("ROI saved successfully.")
