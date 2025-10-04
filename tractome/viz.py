import logging

import numpy as np

from fury import actor
from fury.lib import Group


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

    vertices = mesh_obj.vertices * 10
    faces = mesh_obj.faces

    texture_coords = None
    if texture and hasattr(mesh_obj.visual, "uv"):
        texture_coords = mesh_obj.visual.uv
        logging.info(
            "Flipping the texture coordinates vertically. To move to the"
            " top-left origin."
        )
        texture_coords[:, 1] = 1 - texture_coords[:, 1]

    normals = None
    if hasattr(mesh_obj, "vertex_normals"):
        normals = mesh_obj.vertex_normals

    mesh = actor.surface(
        vertices,
        faces,
        texture=texture,
        texture_coords=texture_coords,
        normals=normals,
    )
    return mesh


def create_streamlines_projection(streamlines, colors, slice_values):
    z_projection = actor.line_projection(
        streamlines,
        plane=(0, 0, -1, slice_values[2]),
        colors=colors,
        thickness=4,
        outline_thickness=0.5,
        lift=-4.0,
    )
    y_projection = actor.line_projection(
        streamlines,
        plane=(0, -1, 0, slice_values[1]),
        colors=colors,
        thickness=4,
        outline_thickness=0.5,
        lift=-4.0,
    )
    x_projection = actor.line_projection(
        streamlines,
        plane=(-1, 0, 0, slice_values[0]),
        colors=colors,
        thickness=4,
        outline_thickness=0.5,
        lift=-4.0,
    )

    obj = Group()
    obj.add(x_projection, y_projection, z_projection)
    return obj


def create_streamlines(streamlines, color):
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
    bundle = actor.streamlines(
        streamlines,
        colors=color,
        thickness=4,
        outline_thickness=1,
        outline_color=(0, 0, 0),
    )
    return bundle


def _deselect_streamtube(streamtube):
    """Deselect a streamtube by setting its opacity to 0.5.

    Parameters
    ----------
    streamtube : Actor
        The streamtube actor to deselect.
    """
    streamtube.material.opacity = 0.5
    streamtube.material.uniform_buffer.update_full()


def _select_streamtube(streamtube):
    """Select a streamtube by setting its opacity to 1.0.

    Parameters
    ----------
    streamtube : Actor
        The streamtube actor to select.
    """
    streamtube.material.opacity = 1.0
    streamtube.material.uniform_buffer.update_full()


def _toggle_streamtube_selection(streamtube):
    """Toggle the selection state of a streamtube.

    Parameters
    ----------
    streamtube : Actor
        The streamtube actor to toggle.
    """
    opacity = streamtube.material.opacity
    if opacity == 1.0:
        _deselect_streamtube(streamtube)
    else:
        _select_streamtube(streamtube)


def toggle_streamtube_selection(event):
    """Handle click events on streamtubes.

    Parameters
    ----------
    event : Event
        The click event.
    """

    st = event.target
    _toggle_streamtube_selection(st)


def create_streamtube(clusters, streamlines):
    """Create streamtubes with radius scaled by number of streamlines in each cluster.

    Parameters
    ----------
    clusters : dict
        Dictionary mapping representative streamline indices to lists of
        streamline indices.
    streamlines : list
        List of streamlines.

    Returns
    -------
    dict
        Dictionary of streamtube actors with scaled radii.
    """
    if not clusters:
        return {}

    cluster_sizes = [len(lines) for lines in clusters.values()]

    min_size = min(cluster_sizes)
    max_size = max(cluster_sizes)

    size_range = max_size - min_size if max_size > min_size else 1

    streamtubes = {}
    for rep, lines in clusters.items():
        num_streamlines = len(lines)
        scaled_radius = ((num_streamlines - min_size) / size_range) * 2.0

        radius = max(scaled_radius, 0.5)

        streamtube = actor.streamtube(
            [streamlines[rep]],
            colors=np.random.rand(3),
            radius=radius,
        )
        streamtube.rep = rep
        streamtube.material.opacity = 0.5
        streamtube.material.uniform_buffer.update_full()
        streamtube.add_event_handler(toggle_streamtube_selection, "pointer_down")
        streamtubes[rep] = streamtube

    return streamtubes


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
    image = actor.volume_slicer(volume, affine=affine)
    return image
