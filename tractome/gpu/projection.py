"""GPU-accelerated point-to-mesh closest-point projection.

A uniform grid built on CPU drives a wgpu compute shader that snaps each
input point to its closest position on a static triangle mesh. The output
buffer can be the storage buffer backing a pygfx (fury v2) Points actor,
so the snapped points are rendered without any CPU<->GPU readback.

Lifecycle
---------
- ``set_mesh(vertices, faces)``      rebuild grid + mesh GPU buffers.
- ``set_points(points)``             allocate input + output GPU buffers.
- ``bind_output_to(actor)``          (optional) reuse a pygfx Points
                                     position buffer as the output.
- ``dispatch(threshold)``            run the compute pass.

The bind group is rebuilt only when one of its resources changes; pure
threshold changes update a single uniform and re-dispatch.
"""

from pathlib import Path

import numpy as np
import wgpu

_SHADER_PATH = Path(__file__).parent / "shaders" / "project_points.wgsl"
_WORKGROUP_SIZE = 64
_UNIFORM_BYTES = 48  # 12 x 4 bytes; matches Uniforms struct in WGSL.


def build_uniform_grid(vertices, faces, cell_size=None):
    """Bin triangles into a uniform 3D grid of axis-aligned cells.

    Parameters
    ----------
    vertices : (V, 3) array_like
    faces : (F, 3) array_like of int
    cell_size : float, optional
        Edge length of one cell. Defaults to ~2x the average mesh edge length.

    Returns
    -------
    bbox_min : (3,) float32
    grid_dims : (3,) uint32        cells per axis
    cell_size : float
    cell_offsets : (n_cells+1,) uint32   CSR-style offsets into tri_indices
    tri_indices : (total,) uint32        triangle ids per cell, concatenated
    """
    vertices = np.ascontiguousarray(vertices, dtype=np.float32)
    faces = np.ascontiguousarray(faces, dtype=np.uint32).reshape(-1, 3)

    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)

    if cell_size is None:
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        avg_edge = (
            np.linalg.norm(v1 - v0, axis=1).mean()
            + np.linalg.norm(v2 - v1, axis=1).mean()
            + np.linalg.norm(v0 - v2, axis=1).mean()
        ) / 3.0
        cell_size = float(avg_edge * 2.0)
    cell_size = max(float(cell_size), 1e-6)

    extent = bbox_max - bbox_min
    grid_dims = np.maximum(1, np.ceil(extent / cell_size)).astype(np.uint32)

    tri_v = vertices[faces]  # (F, 3, 3)
    tri_min = tri_v.min(axis=1)
    tri_max = tri_v.max(axis=1)
    cmin = np.floor((tri_min - bbox_min) / cell_size).astype(np.int64)
    cmax = np.floor((tri_max - bbox_min) / cell_size).astype(np.int64)
    cmin = np.clip(cmin, 0, grid_dims.astype(np.int64) - 1)
    cmax = np.clip(cmax, 0, grid_dims.astype(np.int64) - 1)

    spans = cmax - cmin + 1  # (F, 3)
    n_per_tri = spans.prod(axis=1).astype(np.int64)
    total = int(n_per_tri.sum())

    n_cells = int(grid_dims[0]) * int(grid_dims[1]) * int(grid_dims[2])

    if total == 0:
        return (
            bbox_min.astype(np.float32),
            grid_dims.astype(np.uint32),
            float(cell_size),
            np.zeros(n_cells + 1, dtype=np.uint32),
            np.zeros(0, dtype=np.uint32),
        )

    pair_tri = np.repeat(np.arange(len(faces), dtype=np.int64), n_per_tri)
    starts = np.cumsum(n_per_tri) - n_per_tri
    local = np.arange(total, dtype=np.int64) - np.repeat(starts, n_per_tri)
    sx = spans[pair_tri, 0]
    sy = spans[pair_tri, 1]
    sxy = sx * sy
    lz = local // sxy
    rem = local - lz * sxy
    ly = rem // sx
    lx = rem - ly * sx
    cx = cmin[pair_tri, 0] + lx
    cy = cmin[pair_tri, 1] + ly
    cz = cmin[pair_tri, 2] + lz
    nx = int(grid_dims[0])
    ny = int(grid_dims[1])
    cell_ids = (cz * ny + cy) * nx + cx

    order = np.argsort(cell_ids, kind="stable")
    tri_indices = pair_tri[order].astype(np.uint32, copy=False)
    cell_ids_sorted = cell_ids[order]

    counts = np.bincount(cell_ids_sorted, minlength=n_cells).astype(np.uint32)
    cell_offsets = np.empty(n_cells + 1, dtype=np.uint32)
    cell_offsets[0] = 0
    np.cumsum(counts, out=cell_offsets[1:])

    return (
        bbox_min.astype(np.float32),
        grid_dims.astype(np.uint32),
        float(cell_size),
        cell_offsets,
        tri_indices,
    )


def _pack_uniforms(bbox_min, cell_size, grid_dims, n_points, threshold):
    buf = np.zeros(12, dtype=np.float32)
    buf[0:3] = bbox_min
    buf[3] = cell_size
    # grid_dims and n_points are u32; reinterpret in-place via a view.
    u32_view = buf.view(np.uint32)
    u32_view[4] = int(grid_dims[0])
    u32_view[5] = int(grid_dims[1])
    u32_view[6] = int(grid_dims[2])
    u32_view[7] = int(n_points)
    buf[8] = float(threshold)
    return buf.tobytes()


class PointProjection:
    """Owns the compute pipeline and per-mesh / per-points GPU buffers.

    A single instance can be reused across mesh and point swaps; only the
    affected buffers are rebuilt.
    """

    _STORAGE = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
    _STORAGE_RW = (
        wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
    )
    _UNIFORM = wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST

    def __init__(self, device):
        self.device = device

        shader_src = _SHADER_PATH.read_text()
        self._shader = device.create_shader_module(code=shader_src)
        self._pipeline = device.create_compute_pipeline(
            layout="auto",
            compute={"module": self._shader, "entry_point": "main"},
        )
        self._bind_group_layout = self._pipeline.get_bind_group_layout(0)

        self._uniform_buf = device.create_buffer(
            size=_UNIFORM_BYTES, usage=self._UNIFORM
        )

        self._mesh_buffers = None  # (vertices, faces, cell_offsets, tri_indices)
        self._mesh_meta = None  # (bbox_min, cell_size, grid_dims)
        self._n_points = 0
        self._in_buf = None
        self._out_buf = None
        self._external_out = False  # True if out_buf is borrowed from pygfx
        self._bind_group = None
        self._bind_dirty = True

    # ------------------------------------------------------------------ mesh
    def set_mesh(self, vertices, faces, cell_size=None):
        bbox_min, grid_dims, cell_size, cell_offsets, tri_indices = build_uniform_grid(
            vertices, faces, cell_size=cell_size
        )
        verts_f32 = np.ascontiguousarray(vertices, dtype=np.float32).ravel()
        faces_u32 = np.ascontiguousarray(faces, dtype=np.uint32).ravel()

        d = self.device
        vbuf = d.create_buffer_with_data(data=verts_f32.tobytes(), usage=self._STORAGE)
        fbuf = d.create_buffer_with_data(data=faces_u32.tobytes(), usage=self._STORAGE)
        obuf = d.create_buffer_with_data(
            data=cell_offsets.tobytes(), usage=self._STORAGE
        )
        ibuf = d.create_buffer_with_data(
            data=tri_indices.tobytes(), usage=self._STORAGE
        )

        self._mesh_buffers = (vbuf, fbuf, obuf, ibuf)
        self._mesh_meta = (bbox_min, cell_size, grid_dims)
        self._bind_dirty = True

    # ---------------------------------------------------------------- points
    def set_points(self, points):
        pts = np.ascontiguousarray(points, dtype=np.float32).reshape(-1, 3)
        n = len(pts)
        size = max(n * 3 * 4, 4)

        d = self.device
        self._in_buf = d.create_buffer_with_data(
            data=pts.tobytes(), usage=self._STORAGE
        )

        if not self._external_out or self._out_buf is None:
            self._out_buf = d.create_buffer(size=size, usage=self._STORAGE_RW)
            self._external_out = False

        self._n_points = n
        self._bind_dirty = True

    def update_points(self, points):
        """Fast path when count hasn't changed: just rewrite the input buffer."""
        pts = np.ascontiguousarray(points, dtype=np.float32).reshape(-1, 3)
        if self._in_buf is None or len(pts) != self._n_points:
            self.set_points(pts)
            return
        self.device.queue.write_buffer(self._in_buf, 0, pts.tobytes())

    # ---------------------------------------------------------------- output
    def bind_output_buffer(self, wgpu_buffer):
        """Use an externally-owned wgpu.Buffer as the snapped-points output.

        The buffer must have ``BufferUsage.STORAGE`` set and be sized for
        ``n_points * 3 * 4`` bytes.
        """
        self._out_buf = wgpu_buffer
        self._external_out = True
        self._bind_dirty = True

    @staticmethod
    def prepare_actor(points_actor):
        """Mark the actor's position buffer as STORAGE so it can be a compute target.

        Must be called *before* the buffer is materialized (i.e., before the
        first render). After the first render, ``bind_output_to_actor`` can
        retrieve the live wgpu.Buffer and bind it.
        """
        positions = points_actor.geometry.positions
        positions._wgpu_usage = (
            getattr(positions, "_wgpu_usage", 0)
            | wgpu.BufferUsage.STORAGE
            | wgpu.BufferUsage.COPY_DST
        )

    def bind_output_to_actor(self, points_actor):
        """Bind the actor's position buffer as our output, materializing if needed."""
        positions = points_actor.geometry.positions
        wgpu_buf = getattr(positions, "_wgpu_object", None)
        if wgpu_buf is None:
            # Force pygfx to allocate the GPU buffer right now (with our STORAGE
            # flag already OR'd in by prepare_actor) instead of waiting for the
            # first render to do it lazily.
            from pygfx.renderers.wgpu.engine.update import ensure_wgpu_object

            wgpu_buf = ensure_wgpu_object(positions)
        self.bind_output_buffer(wgpu_buf)

    # -------------------------------------------------------------- dispatch
    def _rebuild_bind_group(self):
        if self._mesh_buffers is None:
            raise RuntimeError("set_mesh() must be called before dispatch().")
        if self._in_buf is None or self._out_buf is None:
            raise RuntimeError("set_points() must be called before dispatch().")

        vbuf, fbuf, obuf, ibuf = self._mesh_buffers
        self._bind_group = self.device.create_bind_group(
            layout=self._bind_group_layout,
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self._in_buf,
                        "offset": 0,
                        "size": self._in_buf.size,
                    },
                },
                {
                    "binding": 1,
                    "resource": {"buffer": vbuf, "offset": 0, "size": vbuf.size},
                },
                {
                    "binding": 2,
                    "resource": {"buffer": fbuf, "offset": 0, "size": fbuf.size},
                },
                {
                    "binding": 3,
                    "resource": {"buffer": obuf, "offset": 0, "size": obuf.size},
                },
                {
                    "binding": 4,
                    "resource": {"buffer": ibuf, "offset": 0, "size": ibuf.size},
                },
                {
                    "binding": 5,
                    "resource": {
                        "buffer": self._uniform_buf,
                        "offset": 0,
                        "size": _UNIFORM_BYTES,
                    },
                },
                {
                    "binding": 6,
                    "resource": {
                        "buffer": self._out_buf,
                        "offset": 0,
                        "size": self._out_buf.size,
                    },
                },
            ],
        )
        self._bind_dirty = False

    def dispatch(self, threshold):
        if self._bind_dirty:
            self._rebuild_bind_group()

        bbox_min, cell_size, grid_dims = self._mesh_meta
        self.device.queue.write_buffer(
            self._uniform_buf,
            0,
            _pack_uniforms(bbox_min, cell_size, grid_dims, self._n_points, threshold),
        )

        encoder = self.device.create_command_encoder()
        cpass = encoder.begin_compute_pass()
        cpass.set_pipeline(self._pipeline)
        cpass.set_bind_group(0, self._bind_group)
        n_groups = (self._n_points + _WORKGROUP_SIZE - 1) // _WORKGROUP_SIZE
        cpass.dispatch_workgroups(max(n_groups, 1))
        cpass.end()
        self.device.queue.submit([encoder.finish()])

    # --------------------------------------------------------------- helpers
    def read_output(self):
        """Read the snapped points back to CPU. Useful for tests/debugging."""
        if self._out_buf is None:
            return None
        raw = self.device.queue.read_buffer(self._out_buf)
        return (
            np.frombuffer(raw, dtype=np.float32).reshape(-1, 3)[: self._n_points].copy()
        )
