import numpy as np

from fury import actor
from fury.colormap import distinguishable_colormap


def create_mesh(mesh_obj, *, texture=None):
    """Create a 3D mesh from the provided mesh object.

    Parameters
    ----------
    mesh_obj : trimesh.Trimesh
        The input mesh object to be converted.
    texture : Optional[Texture], optional
        The texture to be applied to the mesh, by default None

    Returns
    -------
    Mesh
        The created 3D mesh.
    """

    # TODO: remove muliplication by 10e5 when data is fixed.
    vertices = mesh_obj.vertices * 10e5
    faces = mesh_obj.faces

    mesh = actor.surface(vertices, faces, material="basic", texture=texture)
    return mesh


def create_tractogram(sft):
    """Create a 3D tractogram from the provided StatefulTractogram.

    Parameters
    ----------
    sft : StatefulTractogram
        The input tractogram to be converted.

    Returns
    -------
    Line
        The created 3D tractogram.
    """

    streamlines = sft.streamlines
    colors = np.zeros((len(streamlines), 3))
    color_generator = distinguishable_colormap()
    colors = np.tile(next(color_generator), (len(streamlines), 1))
    tractogram = actor.line(streamlines, colors=colors)
    return tractogram


def create_image_slicer(volume, *, affine=None):
    """Create a 2D image from the provided NIfTI file.

    Parameters
    ----------
    volume : ndarray
        The input image data.
    affine : ndarray
        The affine transformation matrix.

    Returns
    -------
    Group
        The created 3D image slicer.
    """
    if affine is None:
        affine = np.eye(4)
    image = actor.slicer(volume)
    return image
