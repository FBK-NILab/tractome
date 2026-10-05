import csv
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import shlex

from dipy.io.image import load_nifti, save_nifti
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import load_tractogram, save_tractogram as dipy_save_tractogram
import numpy as np
import polyxios as px

from fury.colormap import normalize_colors


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
    colors: np.ndarray | None = None


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


def _read_obj_material_colors(path) -> dict[str, tuple[float, float, float]]:
    """Read standard diffuse colors; unavailable colors do not invalidate geometry."""
    colors = {}
    material = None
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line_number, line in enumerate(fh, 1):
                parts = line.partition("#")[0].split()
                if not parts:
                    continue
                if parts[0] == "newmtl":
                    material = " ".join(parts[1:])
                    colors.pop(material, None)
                elif parts[0] == "Kd":
                    try:
                        rgb = tuple(float(value) for value in parts[1:])
                        if len(rgb) != 3 or not all(
                            np.isfinite(value) and 0 <= value <= 1 for value in rgb
                        ):
                            raise ValueError
                    except ValueError:
                        logging.warning(
                            "Ignoring invalid MTL diffuse color in %s at line %d",
                            path,
                            line_number,
                        )
                        continue
                    if material is not None:
                        colors[material] = rgb
    except OSError as error:
        logging.warning("Unable to read OBJ material library %s: %s", path, error)
    return colors


def _read_obj_mesh(path) -> MeshData:
    """Decode OBJ face corners, splitting UV, normal and diffuse-color boundaries."""
    positions = []
    vertex_colors = []
    texcoords = []
    normals = []
    faces_raw = []
    material_colors = {}
    material = None

    def face_index(value, count, token, line_number):
        try:
            index = int(value)
            if index == 0 or not -count <= index <= count:
                raise ValueError
        except ValueError:
            raise ValueError(
                f"Invalid OBJ face index in {path} at line {line_number}: {token}"
            ) from None
        return index - 1 if index > 0 else count + index

    with open(path, encoding="utf-8", errors="replace") as fh:
        for line_number, line in enumerate(fh, 1):
            line = line.partition("#")[0].strip()
            parts = line.split()
            if not parts:
                continue
            directive = parts[0].lower()
            if directive == "v":
                positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
                rgb = None
                if len(parts) in (7, 8):
                    try:
                        rgb = tuple(float(value) for value in parts[4:7])
                        if not all(
                            np.isfinite(value) and 0 <= value <= 255 for value in rgb
                        ):
                            raise ValueError
                    except ValueError:
                        rgb = None
                        logging.warning(
                            "Ignoring invalid OBJ vertex color in %s at line %d",
                            path,
                            line_number,
                        )
                vertex_colors.append(rgb)
            elif directive == "vt":
                texcoords.append(
                    [float(parts[1]), float(parts[2]) if len(parts) > 2 else 0.0]
                )
            elif directive == "vn":
                normals.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif directive == "mtllib":
                for library in shlex.split(line.split(maxsplit=1)[1], posix=False):
                    if (
                        len(library) >= 2
                        and library[0] == library[-1]
                        and library[0] in "\"'"
                    ):
                        library = library[1:-1]
                    material_colors.update(
                        _read_obj_material_colors(Path(path).parent / library)
                    )
            elif directive == "usemtl":
                material = " ".join(parts[1:])
            elif directive == "f":
                face = []
                for token in parts[1:]:
                    corner = token.split("/")
                    vi = face_index(corner[0], len(positions), token, line_number)
                    vti = (
                        face_index(corner[1], len(texcoords), token, line_number)
                        if len(corner) >= 2 and corner[1]
                        else None
                    )
                    vni = (
                        face_index(corner[2], len(normals), token, line_number)
                        if len(corner) >= 3 and corner[2]
                        else None
                    )
                    face.append((vi, vti, vni))
                faces_raw.append((face, material))

    colored_indices = [i for i, rgb in enumerate(vertex_colors) if rgb is not None]
    if colored_indices:
        normalized = normalize_colors([vertex_colors[i] for i in colored_indices])
        for index, rgb in zip(colored_indices, normalized, strict=True):
            vertex_colors[index] = tuple(rgb)

    def color_array(colors):
        if not any(rgb is not None for rgb in colors):
            return None
        if any(rgb is None for rgb in colors):
            logging.warning(
                "OBJ %s contains uncolored face corners; using neutral gray", path
            )
        return np.asarray(
            [rgb if rgb is not None else (0.7, 0.7, 0.7) for rgb in colors],
            dtype=np.float32,
        ).reshape(-1, 3)

    if not faces_raw:
        return MeshData(
            vertices=np.asarray(positions, dtype=np.float32).reshape(-1, 3),
            faces=np.empty((0, 3), dtype=np.int32),
            colors=color_array(vertex_colors),
        )

    unique = {}
    new_pos = []
    new_uv = []
    new_nrm = []
    new_colors = []
    tri_faces = []
    complete_normals = True
    for face, face_material in faces_raw:
        indices = []
        for vi, vti, vni in face:
            rgb = vertex_colors[vi]
            if rgb is None:
                rgb = material_colors.get(face_material)
            key = (vi, vti, vni, rgb)
            if key not in unique:
                unique[key] = len(new_pos)
                new_pos.append(positions[vi])
                new_colors.append(rgb)
                new_uv.append(texcoords[vti] if vti is not None else [0.0, 0.0])
                if vni is None:
                    complete_normals = False
                else:
                    new_nrm.append(normals[vni])
            indices.append(unique[key])
        for i in range(1, len(indices) - 1):
            tri_faces.append([indices[0], indices[i], indices[i + 1]])

    return MeshData(
        vertices=np.asarray(new_pos, dtype=np.float32).reshape(-1, 3),
        faces=np.asarray(tri_faces, dtype=np.int32).reshape(-1, 3),
        normals=(
            np.asarray(new_nrm, dtype=np.float32).reshape(-1, 3)
            if complete_normals and new_nrm
            else None
        ),
        texcoords=(
            np.asarray(new_uv, dtype=np.float32).reshape(-1, 2) if texcoords else None
        ),
        colors=color_array(new_colors),
    )


def read_mesh(file_path, *, texture=None):
    """Read OBJ geometry/colors directly, or other mesh formats using polyxios.

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

    if validated_path.lower().endswith(".obj"):
        mesh = _read_obj_mesh(validated_path)
    else:
        poly = px.read(validated_path)
        vertices = np.asarray(poly.vertices, dtype=np.float32)
        faces = poly.faces
        if faces is not None:
            faces = np.asarray(faces, dtype=np.int32)
        normals = poly.vertex_attrs.get("normals")
        if normals is not None:
            normals = np.asarray(normals, dtype=np.float32)
        texcoords = (
            _extract_texcoords_from_attrs(poly.vertex_attrs) if texture else None
        )
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
    """Save positive ROI voxels as binary uint8 on the ROI's own grid.

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

    roi_uint8 = (np.asarray(roi) > 0).astype(np.uint8)
    save_nifti(validated_path, roi_uint8, affine, dtype=np.uint8)
    logging.info("ROI saved successfully.")
