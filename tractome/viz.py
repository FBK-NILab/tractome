import logging

import numpy as np

from fury import actor


def create_keystroke_card():
    """Create a card displaying the keystroke shortcuts.

    Returns
    -------
    Group
        The group containing the keystroke card and its labels.
    """
    group = actor.Group()
    card = actor.square(
        np.asarray([(70, 90, 0.2)], dtype=np.float32),
        colors=(0.1, 0.1, 0.1),
        scales=(140, 185, 1),
        material="basic",
    )

    left = 10
    y_start = 170
    y_step = -15
    keystrokes = [
        "Key Strokes",
        "a: Select All",
        "n: Select None",
        "i : Swap Selection",
        "d: Delete Selection",
        "e: Expand Selection",
        "c: Collapse Selection",
        "s: Show Selection",
        "h: Hide Selection",
        "x: Toggle this message",
        "t: Save current state",
    ]
    for i, text in enumerate(keystrokes):
        y = y_start + y_step * i
        txt_actor = actor.text(
            text, position=(left, y, 0), font_size=12.0, anchor="top-left"
        )
        txt_actor.local.position = (left, y, 0.1)
        group.add(txt_actor)
    group.add(card)
    return group


def create_roi(roi_data, *, affine=None, color=(1, 0, 0)):
    """Create a 3D ROI from the provided ROI data.

    Parameters
    ----------
    roi_data : ndarray
        The input ROI data.
    affine : ndarray, optional
        The affine transformation matrix.
    color : tuple, optional
        The color of the ROI.

    Returns
    -------
    Mesh
        The created 3D ROI.
    """
    roi = actor.contour_from_roi(
        roi_data, affine=affine, color=color, opacity=0.8, material="phong"
    )
    for contour in roi.children:
        contour.material.alpha_mode = "auto"
        contour.material.depth_write = True
        contour.render_order = 2
    return roi


def create_mesh(mesh_obj, *, texture=None, photographic=True):
    """Create a 3D mesh from the provided mesh object.

    Parameters
    ----------
    mesh_obj : trimesh.Trimesh
        The input mesh object to be converted.
    texture : str or None, optional
        Path to the texture image for the mesh.
    photographic : bool, optional
        When True (default) use basic shading suitable for textured photographic
        rendering; when False use phong shading with vertex normals.

    Returns
    -------
    Mesh
        The created 3D mesh.
    """
    vertices = mesh_obj.vertices * 1
    faces = mesh_obj.faces

    texture_coords = None
    if texture and hasattr(mesh_obj.visual, "uv"):
        uvs = np.asarray(mesh_obj.visual.uv, dtype=np.float32).copy()
        logging.info("Flipping texture coordinates vertically (top-left image origin).")
        uvs[:, 1] = 1.0 - uvs[:, 1]
        texture_coords = uvs

    normals = None
    if hasattr(mesh_obj, "vertex_normals"):
        normals = mesh_obj.vertex_normals

    mesh = actor.surface(
        vertices,
        faces,
        material="basic" if photographic else "phong",
        texture=texture,
        texture_coords=texture_coords,
        normals=normals,
    )
    mesh.material.alpha_mode = "solid"
    mesh.material.depth_write = True
    mesh.render_order = 2
    return mesh


def create_streamlines_projection(streamlines, colors, slice_values):
    """Project a streamline bundle onto each principal slice plane.

    Parameters
    ----------
    streamlines : Sequence[ndarray]
        The streamlines to project.
    colors : tuple or ndarray
        Per-line colour shared by all three projections.
    slice_values : tuple of int
        Current ``(x, y, z)`` slice indices used to position each plane.

    Returns
    -------
    Group
        Group actor holding the X, Y, and Z plane projections.
    """
    thickness = 3
    outline_thickness = 1.0
    z_projection = actor.line_projection(
        streamlines,
        plane=(0, 0, -1, slice_values[2]),
        colors=colors,
        thickness=thickness,
        outline_thickness=outline_thickness,
        lift=-4.0,
    )
    y_projection = actor.line_projection(
        streamlines,
        plane=(0, -1, 0, slice_values[1]),
        colors=colors,
        thickness=thickness,
        outline_thickness=outline_thickness,
        lift=-4.0,
    )
    x_projection = actor.line_projection(
        streamlines,
        plane=(-1, 0, 0, slice_values[0]),
        colors=colors,
        thickness=thickness,
        outline_thickness=outline_thickness,
        lift=-4.0,
    )

    obj = actor.Group()
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
        thickness=1,
        outline_thickness=1,
        outline_color=(0, 0, 0),
    )
    bundle.material.alpha_mode = "auto"
    bundle.material.depth_write = True
    bundle.render_order = 1
    return bundle


def create_parcels(pts, colors):
    """Create point based representation of parcels

    Parameters
    ----------
    pts : ndarray
        The input parcel points.
    colors : ndarray
        The input parcel colors.

    Returns
    -------
    Point
        The created point-based representation of parcels.
    """
    colors = np.asarray(colors, dtype=np.float32) / 255.0
    pts = np.asarray(pts, dtype=np.float32) * 1
    parcels = actor.point(pts, colors=colors, size=4.0, enable_picking=False)
    return parcels


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


def create_streamtube(line, color, radius):
    """Create a streamtube from a line.

    Parameters
    ----------
    line : ndarray
        The input line.
    color : tuple
        The color of the streamtube.
    radius : float
        The radius of the streamtube.

    Returns
    -------
    Streamtube
        The created streamtube.
    """

    streamtube = actor.streamtube(
        [line],
        colors=color,
        radius=radius,
        backend="cpu",
    )
    streamtube.material.opacity = 0.5
    streamtube.material.alpha_mode = "auto"
    streamtube.material.depth_write = True
    streamtube.render_order = 1
    streamtube.material.uniform_buffer.update_full()
    streamtube.add_event_handler(toggle_streamtube_selection, "on_selection")
    return streamtube


def _voxel_bbox_for_world_box(shape, inv_affine, world_corners):
    """Voxel-space (vmin, vmax) bbox covering the given world-space corners.

    The corners are transformed to voxel space and clipped to the volume
    bounds so callers can iterate only the relevant subgrid.
    """
    homogeneous = np.hstack(
        [world_corners, np.ones((world_corners.shape[0], 1), dtype=np.float64)]
    )
    voxel_corners = (inv_affine @ homogeneous.T).T[:, :3]
    vmin = np.maximum(np.floor(voxel_corners.min(axis=0)).astype(int), 0)
    vmax = np.minimum(
        np.ceil(voxel_corners.max(axis=0)).astype(int) + 1,
        np.asarray(shape, dtype=int),
    )
    return vmin, vmax


def rasterize_sphere(shape, affine, world_center, world_radius):
    """Rasterize a sphere into a binary voxel volume.

    Parameters
    ----------
    shape : tuple of int
        Output volume shape ``(nx, ny, nz)``.
    affine : ndarray
        4x4 voxel-to-world affine.
    world_center : array-like of 3 floats
        Sphere center in world coordinates.
    world_radius : float
        Sphere radius in world units.

    Returns
    -------
    ndarray
        ``uint8`` binary volume with voxels inside the sphere set to 1.
    """
    inv_affine = np.linalg.inv(affine)
    cx, cy, cz = world_center
    r = float(world_radius)
    corners = np.array(
        [
            [cx + sx * r, cy + sy * r, cz + sz * r]
            for sx in (-1, 1)
            for sy in (-1, 1)
            for sz in (-1, 1)
        ],
        dtype=np.float64,
    )
    vmin, vmax = _voxel_bbox_for_world_box(shape, inv_affine, corners)
    volume = np.zeros(shape, dtype=np.uint8)
    if np.any(vmin >= vmax):
        return volume

    xs = np.arange(vmin[0], vmax[0])
    ys = np.arange(vmin[1], vmax[1])
    zs = np.arange(vmin[2], vmax[2])
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack(
        [gx.ravel(), gy.ravel(), gz.ravel(), np.ones(gx.size, dtype=np.float64)],
        axis=1,
    )
    world_pts = (affine @ pts.T).T[:, :3]
    dists = np.linalg.norm(
        world_pts - np.asarray(world_center, dtype=np.float64), axis=1
    )
    mask = dists <= r
    volume[gx.ravel(), gy.ravel(), gz.ravel()] = mask.astype(np.uint8)
    return volume


def rasterize_box(shape, affine, world_corner_a, world_corner_b, axis, world_depth):
    """Rasterize an axis-aligned rectangular slab into a binary voxel volume.

    Parameters
    ----------
    shape : tuple of int
        Output volume shape ``(nx, ny, nz)``.
    affine : ndarray
        4x4 voxel-to-world affine.
    world_corner_a, world_corner_b : array-like of 3 floats
        Two opposite corners of the rectangle's diagonal in world coordinates.
        Both points lie on the slice plane; the slab spans their bounding
        box in the two axes perpendicular to ``axis``.
    axis : int
        World axis perpendicular to the slab (0, 1, or 2).
    world_depth : float
        Total thickness along ``axis``, centred on the midpoint of the
        two corners along that axis. Pass one voxel of spacing for a
        single-voxel-deep slab.

    Returns
    -------
    ndarray
        ``uint8`` binary volume with voxels inside the slab set to 1.
    """
    inv_affine = np.linalg.inv(affine)
    a = np.asarray(world_corner_a, dtype=np.float64)
    b = np.asarray(world_corner_b, dtype=np.float64)
    half_d = float(world_depth) / 2.0
    lo = np.minimum(a, b).copy()
    hi = np.maximum(a, b).copy()
    center_axis = (a[axis] + b[axis]) / 2.0
    lo[axis] = center_axis - half_d
    hi[axis] = center_axis + half_d

    corners = np.array(
        [
            [
                lo[0] if sx == 0 else hi[0],
                lo[1] if sy == 0 else hi[1],
                lo[2] if sz == 0 else hi[2],
            ]
            for sx in (0, 1)
            for sy in (0, 1)
            for sz in (0, 1)
        ],
        dtype=np.float64,
    )
    vmin, vmax = _voxel_bbox_for_world_box(shape, inv_affine, corners)
    volume = np.zeros(shape, dtype=np.uint8)
    if np.any(vmin >= vmax):
        return volume

    xs = np.arange(vmin[0], vmax[0])
    ys = np.arange(vmin[1], vmax[1])
    zs = np.arange(vmin[2], vmax[2])
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack(
        [gx.ravel(), gy.ravel(), gz.ravel(), np.ones(gx.size, dtype=np.float64)],
        axis=1,
    )
    world_pts = (affine @ pts.T).T[:, :3]
    mask = (
        (world_pts[:, 0] >= lo[0])
        & (world_pts[:, 0] <= hi[0])
        & (world_pts[:, 1] >= lo[1])
        & (world_pts[:, 1] <= hi[1])
        & (world_pts[:, 2] >= lo[2])
        & (world_pts[:, 2] <= hi[2])
    )
    volume[gx.ravel(), gy.ravel(), gz.ravel()] = mask.astype(np.uint8)
    return volume


def create_image_slicer(volume, *, affine=None, mode="auto", depth_write=True):
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
    image = actor.volume_slicer(
        volume, affine=affine, alpha_mode=mode, depth_write=depth_write
    )
    for plane in image.children:
        plane.material.alpha_mode = "auto"
        plane.render_order = 0
    return image
