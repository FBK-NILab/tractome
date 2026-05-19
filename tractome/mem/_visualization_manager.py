import os

from dipy.tracking.distances import bundles_distances_mam
import numpy as np

from fury import actor as _fury_actor, distinguishable_colormap
from fury.actor import set_group_visibility, show_slices
from tractome.compute import (
    calculate_filter,
    compute_dissimilarity,
    filter_streamline_ids,
    mkbm_clustering,
    transform_roi_to_world_grid,
)
from tractome.gpu import PointProjection
from tractome.mem import ClusterState, input_manager, state_manager
from tractome.viz import (
    create_image_slicer,
    create_mesh,
    create_parcels,
    create_roi,
    create_streamlines,
    create_streamlines_projection,
    create_streamtube,
)


class VisualizationManager:
    """A class to manage the visualization of the inputs."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        """Create a new instance of the VisualizationManager if one does not exist.

        Parameters
        ----------
        *args : tuple
            Variable length argument list.
        **kwargs : dict
            Arbitrary keyword arguments.

        Returns
        -------
        VisualizationManager
            The instance of the VisualizationManager.
        """
        if not cls._instance:
            cls._instance = super(VisualizationManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self._visualizations = {
            "tractogram": None,
            "t1": None,
            "mesh": None,
            "mesh_projection": None,
            "roi": [],
            "parcel": None,
        }
        self._2d_visualizations = {
            "t1": None,
            "roi": [],
            "tractogram": None,
        }
        self._roi_colormap = distinguishable_colormap()
        self._roi_colors = {}
        self._roi_visibility = {}
        self._roi_applied = {}
        self._roi_negated = {}
        self._wgpu_device = None
        self._mesh_projection = None
        self._mesh_projection_bound = False
        self._mesh_projection_n_points = 0

    def visualize_t1(self):
        """Visualize the T1 image.

        Returns
        -------
        Group
            The visualized T1 image with X, Y, and Z slices.
        """
        if not input_manager.has_t1:
            return None

        img, affine, _, _ = input_manager.get_current_t1()
        self._visualizations["t1"] = [create_image_slicer(img, affine=affine)]
        return self._visualizations["t1"]

    def visualize_t1_2d(self):
        """Build the 2D T1 image slicer (weighted-blend, depth-write off).

        Returns
        -------
        list | None
            A single-element list with the 2D image slicer actor, or None
            when no T1 image is loaded.
        """
        if not input_manager.has_t1:
            self._2d_visualizations["t1"] = None
            return None

        img, affine, _, _ = input_manager.get_current_t1()
        slicer = create_image_slicer(
            img, affine=affine, mode="weighted_blend", depth_write=False
        )
        set_group_visibility(slicer, state_manager.t1_slice_visibility_2d)
        show_slices(slicer, state_manager.t1_state)
        self._2d_visualizations["t1"] = [slicer]
        return self._2d_visualizations["t1"]

    def visualize_rois_2d(self):
        """Build per-ROI 2D RGBA slicers matching the current ROI list.

        Each ROI volume is converted to an RGBA volume coloured with the
        same RGB tuple used by the 3D contour, then wrapped in a slicer
        configured for translucent overlay rendering.

        Returns
        -------
        list
            The list of 2D ROI slicer actors. Empty when no ROI is
            provided.
        """
        self._2d_visualizations["roi"] = []
        if not input_manager.has_roi:
            return self._2d_visualizations["roi"]

        for index in range(len(input_manager.provided_roi_paths)):
            roi_volume, affine, path, _ = input_manager.get_roi_at(index)
            color = self._roi_colors.get(path)
            if color is None:
                color = next(self._roi_colormap)
                self._roi_colors[path] = color
            rgba = self._build_roi_rgba_volume(roi_volume, color)
            slicer = create_image_slicer(
                rgba, affine=affine, mode="weighted_blend", depth_write=False
            )
            for child in slicer.children:
                child.material.opacity = 0.3
            set_group_visibility(slicer, state_manager.t1_slice_visibility_2d)
            show_slices(slicer, np.asarray(state_manager.t1_state) + 0.1)
            slicer.visible = self._roi_visibility.get(path, True)
            self._2d_visualizations["roi"].append(slicer)
        return self._2d_visualizations["roi"]

    def visualize_streamlines_projection_2d(self):
        """Build streamline projections for selected clusters in 2D.

        Returns
        -------
        list
            A single-element list with the projection group actor, or an
            empty list when no clusters are selected.
        """
        self._2d_visualizations["tractogram"] = None
        if not input_manager.has_tractogram or not state_manager.has_states():
            return []

        latest_state = state_manager.get_latest_state()
        if latest_state.tractogram_states is None:
            return []

        sft, _, _, _ = input_manager.get_current_tractogram()
        slice_values = state_manager.t1_state

        projections = []
        for state_data in latest_state.tractogram_states.values():
            if not state_data["selected"]:
                continue
            streamlines = [
                np.asarray(sft.streamlines[i]) for i in state_data["streamline_ids"]
            ]
            if not streamlines:
                continue
            projection = create_streamlines_projection(
                streamlines=streamlines,
                colors=state_data["color"],
                slice_values=slice_values,
            )
            set_group_visibility(projection, state_manager.t1_slice_visibility_2d)
            projections.append(projection)

        self._2d_visualizations["tractogram"] = projections
        return projections

    def _build_roi_rgba_volume(self, volume, color):
        """Create an RGBA-style colour volume for a single ROI.

        Parameters
        ----------
        volume : ndarray
            The binary ROI volume.
        color : tuple of float
            RGB tuple in [0, 1] applied to occupied voxels.

        Returns
        -------
        ndarray
            Volume of shape ``(*volume.shape, 3)`` containing the colour.
        """
        rgba = np.zeros((*volume.shape, 3), dtype=np.float32)
        mask = volume != 0
        if np.any(mask):
            rgba[mask, :3] = color
        return rgba

    @property
    def t1_2d_visualizations(self):
        """Return the 2D T1 visualization list."""
        return self._2d_visualizations["t1"]

    @property
    def roi_2d_visualizations(self):
        """Return the 2D ROI visualization list."""
        return self._2d_visualizations["roi"]

    @property
    def streamlines_2d_visualizations(self):
        """Return the 2D streamline projection list."""
        return self._2d_visualizations["tractogram"] or []

    def visualize_mesh(self):
        """Build the mesh actor for the current mesh/texture pair.

        Returns
        -------
        list | None
            A single-element list containing the mesh actor, or None if no mesh.
        """
        if not input_manager.has_mesh:
            self._visualizations["mesh"] = None
            return None

        mesh_obj, texture_path, _, _ = input_manager.get_current_mesh()
        mesh_actor = create_mesh(
            mesh_obj,
            texture=texture_path,
            photographic=state_manager.mesh_photographic,
        )
        mesh_actor.visible = state_manager.mesh_visible
        opacity = state_manager.mesh_opacity / 100.0
        mesh_actor.material.opacity = opacity
        if opacity < 1.0:
            mesh_actor.material.alpha_mode = "blend"
            mesh_actor.material.depth_write = False
        else:
            mesh_actor.material.alpha_mode = "solid"
            mesh_actor.material.depth_write = True
        self._visualizations["mesh"] = [mesh_actor]
        return self._visualizations["mesh"]

    def toggle_mesh_visibility(self):
        """Toggle mesh actor visibility."""
        mesh = self._visualizations["mesh"]
        if not mesh:
            return
        mesh[0].visible = not mesh[0].visible
        state_manager.mesh_visible = mesh[0].visible

    def set_mesh_opacity(self, value):
        """Set mesh opacity from a 0–100 slider value."""
        mesh = self._visualizations["mesh"]
        if not mesh:
            return
        state_manager.mesh_opacity = value
        opacity = value / 100.0
        mesh[0].material.opacity = opacity
        if opacity < 1.0:
            mesh[0].material.alpha_mode = "blend"
            mesh[0].material.depth_write = False
        else:
            mesh[0].material.alpha_mode = "solid"
            mesh[0].material.depth_write = True

    def sync_mesh_visibility_from_state(self):
        """Apply state_manager.mesh_visible to the mesh actor if present."""
        mesh = self._visualizations["mesh"]
        if not mesh:
            return
        mesh[0].visible = state_manager.mesh_visible

    @property
    def mesh_is_visible(self):
        """Whether the mesh actor is shown (defaults True if absent)."""
        mesh = self._visualizations["mesh"]
        if not mesh:
            return True
        return bool(mesh[0].visible)

    # ------------------------------------------------------------- mesh projection
    def set_wgpu_device(self, device):
        """Register the wgpu device used by the GPU projection pipeline."""
        self._wgpu_device = device

    @property
    def mesh_projection_visualizations(self):
        return self._visualizations["mesh_projection"]

    def _gather_streamline_points(self):
        """Return (points, colors) for streamlines in currently-expanded clusters.

        Only expanded clusters contribute points; collapsed clusters are
        skipped even if visible. Returns ``(None, None)`` if no cluster is
        currently expanded.
        """
        if not input_manager.has_tractogram:
            return None, None
        sft, _, _, _ = input_manager.get_current_tractogram()
        streamlines = sft.streamlines
        if len(streamlines) == 0:
            return None, None

        if not state_manager.has_states():
            return None, None
        tractogram_states = state_manager.get_latest_state().tractogram_states
        if tractogram_states is None:
            return None, None

        color_map = {}
        for state_data in tractogram_states.values():
            if not state_data.get("expanded"):
                continue
            if not state_data.get("visible", True):
                continue
            color = np.asarray(state_data["color"], dtype=np.float32).ravel()[:3]
            for sid in state_data["streamline_ids"]:
                color_map[int(sid)] = color

        if not color_map:
            return None, None

        sids = sorted(color_map)
        selected = [
            np.asarray(streamlines[sid], dtype=np.float32).reshape(-1, 3)
            for sid in sids
        ]
        lengths = np.fromiter(
            (len(s) for s in selected), dtype=np.int64, count=len(selected)
        )
        pts = np.concatenate(selected, axis=0)
        color_arr = np.stack([color_map[sid] for sid in sids], axis=0)
        colors = np.repeat(color_arr, lengths, axis=0)
        return pts, colors

    def visualize_mesh_projection(self):
        """Build the projected-points actor and seed the GPU projection state.

        Returns a single-element list with the actor, or None if prerequisites
        (mesh, tractogram, registered wgpu device) are missing. The caller is
        expected to add the actor to the scene and trigger a render BEFORE
        calling :meth:`update_mesh_projection`, so the actor's GPU position
        buffer is materialized and can be bound as compute output.
        """
        if self._wgpu_device is None:
            return None
        if not input_manager.has_mesh or not input_manager.has_tractogram:
            self._visualizations["mesh_projection"] = None
            return None
        pts, colors = self._gather_streamline_points()
        if pts is None:
            self._visualizations["mesh_projection"] = None
            return None

        mesh_obj, _, _, _ = input_manager.get_current_mesh()

        if self._mesh_projection is None:
            self._mesh_projection = PointProjection(self._wgpu_device)
        self._mesh_projection.set_mesh(
            np.asarray(mesh_obj.vertices, dtype=np.float32),
            np.asarray(mesh_obj.faces, dtype=np.uint32),
        )
        self._mesh_projection.set_points(pts)
        self._mesh_projection_n_points = len(pts)
        self._mesh_projection_bound = False

        proj_actor = _fury_actor.point(
            pts,
            colors=colors,
            size=2.0,
            enable_picking=False,
        )
        PointProjection.prepare_actor(proj_actor)

        self._visualizations["mesh_projection"] = [proj_actor]
        return self._visualizations["mesh_projection"]

    def update_mesh_projection(self, threshold=None):
        """Run the compute pass and write snapped positions into the actor buffer.

        Safe to call any time after :meth:`visualize_mesh_projection` — the
        GPU buffer is materialized eagerly and the CPU<->GPU sync state is
        adjusted to keep the compute output authoritative.
        """
        if self._mesh_projection is None:
            return
        viz = self._visualizations["mesh_projection"]
        if not viz:
            return

        positions = viz[0].geometry.positions
        if not self._mesh_projection_bound:
            self._mesh_projection.bind_output_to_actor(viz[0])
            self._mesh_projection_bound = True

        if threshold is None:
            threshold = state_manager.mesh_projection_threshold
        self._mesh_projection.dispatch(float(threshold))

        # The GPU buffer now holds the snapped positions. Suppress any pending
        # CPU->GPU upload pygfx may have queued from the original streamline
        # data; otherwise the next render would overwrite our compute output.
        if hasattr(positions, "_chunks_dirt_flag"):
            positions._chunks_dirt_flag = 0
            if positions._chunk_mask is not None:
                positions._chunk_mask[:] = False

    def set_mesh_projection_visible(self, visible):
        viz = self._visualizations["mesh_projection"]
        if not viz:
            return
        viz[0].visible = bool(visible)

    def clear_mesh_projection(self):
        """Forget the projection actor and binding (e.g., on mesh/tractogram swap)."""
        self._visualizations["mesh_projection"] = None
        self._mesh_projection_bound = False

    def rebuild_mesh_projection(self):
        """Build a replacement projection actor from the current state.

        Returns ``(old_viz, new_viz)`` — both lists or ``None`` — so the caller
        (which owns the scene) can remove the old actor and add the new one.
        Internal projection bookkeeping is reset; the caller must call
        :meth:`update_mesh_projection` after adding ``new_viz`` to the scene.
        """
        old_viz = self._visualizations["mesh_projection"]
        self._visualizations["mesh_projection"] = None
        self._mesh_projection_bound = False
        new_viz = self.visualize_mesh_projection()
        return old_viz, new_viz

    @property
    def mesh_visualizations(self):
        """Return the mesh visualization list."""
        return self._visualizations["mesh"]

    def visualize_tractogram(self, *, nb_clusters=100):
        """Visualize the tractogram.

        Parameters
        ----------
        nb_clusters : int, optional
            The number of clusters to create.

        Returns
        -------
        list
            The actors representing the tractogram.
        """

        if not input_manager.has_tractogram:
            return None

        sft, _, _, _ = input_manager.get_current_tractogram()
        is_embeddings_present = "dismatrix" in sft.data_per_streamline
        if not is_embeddings_present:
            n_jobs = max(1, (os.cpu_count() or 1) - 2)
            data_dissimilarity = compute_dissimilarity(
                np.asarray(sft.streamlines, dtype=object),
                distance=bundles_distances_mam,
                prototype_policy="sff",
                num_prototypes=40,
                verbose=False,
                size_limit=5000000,
                n_jobs=n_jobs,
            )
            sft.data_per_streamline["dismatrix"] = data_dissimilarity

        if not state_manager.has_states():
            state_manager.add_state(
                ClusterState(nb_clusters, np.arange(len(sft.streamlines)), 1000)
            )
        else:
            state_manager.get_latest_state().nb_clusters = nb_clusters

        self._apply_tractogram_states()

        return self._visualizations["tractogram"]

    def _get_actors(self):
        """Get the actors for the visualization."""
        actors = []
        for state_data in state_manager.get_latest_state().tractogram_states.values():
            if state_data["expanded"] is not True:
                actors.append(state_data["rep_actor"])
            else:
                actors.append(state_data["lines_actor"])
        return actors

    def _apply_tractogram_states(self):
        """Apply the tractogram states to the visualization."""
        sft, _, _, _ = input_manager.get_current_tractogram()
        latest_state = state_manager.get_latest_state()
        if input_manager.has_roi and latest_state.filtered_streamline_ids is None:
            self.apply_roi_filter()
            latest_state = state_manager.get_latest_state()
        if latest_state.tractogram_states is not None:
            for cluster_id, state_data in latest_state.tractogram_states.items():
                if not state_data["expanded"]:
                    if state_data["rep_actor"] is None:
                        state_data["rep_actor"] = self._create_cluster_rep_actor(
                            cluster_id,
                            sft.streamlines[cluster_id],
                            state_data["color"],
                            state_data["radius"],
                        )
                    if state_data["selected"]:
                        state_data["rep_actor"].material.opacity = 1.0
                    else:
                        state_data["rep_actor"].material.opacity = 0.5
                    if state_data["visible"]:
                        state_data["rep_actor"].visible = True
                    else:
                        state_data["rep_actor"].visible = False
                    state_data["lines_actor"] = None
                elif state_data["expanded"] and state_data["lines_actor"] is None:
                    state_data["lines_actor"] = self._create_cluster_lines_actor(
                        cluster_id,
                        sft.streamlines[state_data["streamline_ids"]],
                        state_data["color"],
                    )
                    state_data["rep_actor"] = None
                    if state_data["visible"]:
                        state_data["lines_actor"].visible = True
                    else:
                        state_data["lines_actor"].visible = False
        else:
            self._perform_clustering(sft, latest_state)

        self._visualizations["tractogram"] = self._get_actors()

    def _perform_clustering(self, sft, state):
        """Perform clustering on the tractogram.

        When ``state.filtered_streamline_ids`` is set, clustering is
        restricted to those filtered IDs so that downstream visualization
        only shows streamlines passing the current ROI filter.

        Parameters
        ----------
        sft : StatefulTractogram
            The tractogram to cluster.
        state : ClusterState
            The state to perform clustering on.
        """
        colormap = distinguishable_colormap()
        streamline_ids = state.streamline_ids
        if state.filtered_streamline_ids is not None:
            streamline_ids = np.asarray(
                sorted(
                    set(np.asarray(streamline_ids, dtype=np.int32).tolist())
                    & set(
                        np.asarray(
                            state.filtered_streamline_ids, dtype=np.int32
                        ).tolist()
                    )
                ),
                dtype=np.int32,
            )
        if len(streamline_ids) == 0:
            state.nb_clusters = 0
            state.tractogram_states = {}
            return
        nb_clusters = min(max(1, int(state.nb_clusters)), len(streamline_ids))
        state.nb_clusters = nb_clusters
        clusters = mkbm_clustering(
            sft.data_per_streamline["dismatrix"],
            n_clusters=nb_clusters,
            streamline_ids=streamline_ids,
        )
        min_size = min(len(streamline_ids) for streamline_ids in clusters.values())
        max_size = max(len(streamline_ids) for streamline_ids in clusters.values())
        size_range = max_size - min_size if max_size > min_size else 1
        for cluster_id, streamline_ids in clusters.items():
            num_streamlines = len(streamline_ids)
            scaled_radius = ((num_streamlines - min_size) / size_range) * 2.0
            radius = max(scaled_radius, 1)
            cluster_entry = state_manager.create_cluster_entry(
                cluster_id, streamline_ids, next(colormap), radius
            )
            cluster_entry["rep_actor"] = self._create_cluster_rep_actor(
                cluster_id,
                sft.streamlines[cluster_id],
                cluster_entry["color"],
                cluster_entry["radius"],
            )

    def _create_cluster_rep_actor(self, cluster_id, line, color, radius):
        """Create a representative actor for a cluster."""
        rep_actor = create_streamtube(line, color, radius)
        rep_actor.rep = cluster_id
        rep_actor.add_event_handler(self._toggle_cluster_selection, "on_selection")
        return rep_actor

    def _create_cluster_lines_actor(self, cluster_id, streamlines, color):
        """Create a lines actor for a cluster."""
        lines_actor = create_streamlines(streamlines, color)
        lines_actor.rep = cluster_id
        return lines_actor

    def _toggle_cluster_selection(self, event):
        """Toggle the selection state of a cluster.

        Parameters
        ----------
        event : Event
            The click event.
        """
        cluster = event.target
        state_manager.toggle_cluster_selection(cluster.rep)

    def expand_clusters(self):
        """Expand selected clusters."""
        state_manager.expand_clusters()
        self._apply_tractogram_states()

    def collapse_clusters(self):
        """Collapse selected clusters."""
        state_manager.collapse_clusters()
        self._apply_tractogram_states()

    def show_clusters(self):
        """Show selected clusters."""
        state_manager.show_clusters()
        self._apply_tractogram_states()

    def hide_clusters(self):
        """Hide selected clusters."""
        state_manager.hide_clusters()
        self._apply_tractogram_states()

    def select_all_clusters(self):
        """Select all clusters."""
        state_manager.select_all_clusters()
        self._apply_tractogram_states()

    def select_none_clusters(self):
        """Select none clusters."""
        state_manager.select_none_clusters()
        self._apply_tractogram_states()

    def swap_clusters(self):
        """Swap selected clusters."""
        state_manager.swap_clusters()
        self._apply_tractogram_states()

    def delete_clusters(self):
        """Delete selected clusters."""
        latest_state = state_manager.get_latest_state()
        streamline_ids = []
        for state_data in latest_state.tractogram_states.values():
            if state_data["selected"]:
                streamline_ids.extend(state_data["streamline_ids"])
        streamline_ids = np.unique(streamline_ids)
        nb_clusters = min(latest_state.nb_clusters, len(streamline_ids))
        state_manager.add_state(
            ClusterState(nb_clusters, streamline_ids, latest_state.max_clusters)
        )
        self._apply_tractogram_states()

    def toggle_t1_visibility(self):
        """Toggle the visibility of the T1 image slicer."""
        t1 = self._visualizations["t1"]
        if not t1:
            return
        for actor in t1:
            actor.visible = not actor.visible

    def show_t1_slices(self, x, y, z):
        """Show the T1 slices on every active T1/ROI/projection actor.

        Parameters
        ----------
        x : int
            The X slice index.
        y : int
            The Y slice index.
        z : int
            The Z slice index.
        """
        state_manager.t1_state = [x, y, z]
        if self._visualizations["t1"]:
            show_slices(self._visualizations["t1"][0], state_manager.t1_state)
        if self._2d_visualizations["t1"]:
            show_slices(self._2d_visualizations["t1"][0], state_manager.t1_state)
        for roi_slicer in self._2d_visualizations["roi"]:
            show_slices(roi_slicer, np.asarray(state_manager.t1_state) + 0.1)
        for projection in self.streamlines_2d_visualizations:
            show_slices(projection, state_manager.t1_state)

    def toggle_t1_slice_visibility(self, x, y, z):
        """Toggle the visibility of the T1 slices in 3D mode.

        Parameters
        ----------
        x : int
            Whether the X slice is visible.
        y : int
            Whether the Y slice is visible.
        z : int
            Whether the Z slice is visible.
        """
        if not self._visualizations["t1"]:
            return
        state_manager.t1_slice_visibility = [x, y, z]
        set_group_visibility(
            self._visualizations["t1"][0], state_manager.t1_slice_visibility
        )

    def toggle_t1_slice_visibility_2d(self, x, y, z):
        """Toggle the visibility of the T1 slices in 2D mode.

        Updates the 2D T1 actor as well as all per-ROI 2D slicers and
        streamline projections so that exactly one axis is shown.

        Parameters
        ----------
        x : int
            Whether the X slice is visible.
        y : int
            Whether the Y slice is visible.
        z : int
            Whether the Z slice is visible.
        """
        state_manager.t1_slice_visibility_2d = [x, y, z]
        if self._2d_visualizations["t1"]:
            set_group_visibility(
                self._2d_visualizations["t1"][0],
                state_manager.t1_slice_visibility_2d,
            )
        for roi_slicer in self._2d_visualizations["roi"]:
            set_group_visibility(roi_slicer, state_manager.t1_slice_visibility_2d)
        for projection in self.streamlines_2d_visualizations:
            set_group_visibility(projection, state_manager.t1_slice_visibility_2d)

    @property
    def t1_is_visible(self):
        """Whether the T1 visualization is currently shown (defaults True if absent)."""
        t1 = self._visualizations["t1"]
        if not t1:
            return True
        return bool(t1[0].visible)

    @property
    def tractogram_visualizations(self):
        """Get the tractogram visualizations."""
        return self._visualizations["tractogram"]

    @property
    def t1_visualizations(self):
        """Get the T1 visualizations."""
        return self._visualizations["t1"]

    def visualize_parcel(self):
        """Build the parcel point actor for the current parcel file.

        Returns
        -------
        list | None
            A single-element list containing the parcel actor, or None if no parcel.
        """
        if not input_manager.has_parcel:
            self._visualizations["parcel"] = None
            return None

        points, colors, _, _ = input_manager.get_current_parcel()
        parcel_actor = create_parcels(points, colors)
        parcel_actor.visible = state_manager.parcel_visible
        parcel_actor.material.size = state_manager.parcel_size / 25.0
        self._visualizations["parcel"] = [parcel_actor]
        return self._visualizations["parcel"]

    def toggle_parcel_visibility(self):
        """Toggle parcel actor visibility."""
        parcel = self._visualizations["parcel"]
        if not parcel:
            return
        parcel[0].visible = not parcel[0].visible
        state_manager.parcel_visible = parcel[0].visible

    def set_parcel_size(self, value):
        """Set parcel point size from a 0–100 slider (legacy mapping: value/25)."""
        parcel = self._visualizations["parcel"]
        if not parcel:
            return
        state_manager.parcel_size = value
        parcel[0].material.size = value / 25.0

    def sync_parcel_visibility_from_state(self):
        """Apply state_manager.parcel_visible to the parcel actor if present."""
        parcel = self._visualizations["parcel"]
        if not parcel:
            return
        parcel[0].visible = state_manager.parcel_visible

    @property
    def parcel_is_visible(self):
        """Whether the parcel actor is shown (defaults True if absent)."""
        parcel = self._visualizations["parcel"]
        if not parcel:
            return True
        return bool(parcel[0].visible)

    @property
    def parcel_visualizations(self):
        """Return the parcel visualization list."""
        return self._visualizations["parcel"]

    def visualize_rois(self):
        """Build actors for every provided ROI.

        Colors and per-ROI visibility are cached by file path so they
        remain stable across rebuilds.

        Returns
        -------
        list
            The list of ROI actors. Empty when no ROI is provided.
        """
        self._visualizations["roi"] = []
        if not input_manager.has_roi:
            return self._visualizations["roi"]

        opacity = state_manager.roi_opacity / 100.0
        for index in range(len(input_manager.provided_roi_paths)):
            roi_volume, affine, path, _ = input_manager.get_roi_at(index)
            color = self._roi_colors.get(path)
            if color is None:
                color = next(self._roi_colormap)
                self._roi_colors[path] = color
            roi_actor = create_roi(roi_volume, affine=affine, color=color)
            self._apply_roi_opacity(roi_actor, opacity)
            roi_actor.color = color
            roi_actor.visible = self._roi_visibility.get(path, True)
            self._visualizations["roi"].append(roi_actor)
        return self._visualizations["roi"]

    def _apply_roi_opacity(self, roi_actor, opacity):
        """Apply an opacity value to every child of an ROI Group actor.

        Parameters
        ----------
        roi_actor : Group
            The ROI group whose contour children should be updated.
        opacity : float
            The opacity value in the [0, 1] range.
        """
        for contour in roi_actor.children:
            contour.material.opacity = opacity
            if opacity < 1.0:
                contour.material.alpha_mode = "blend"
                contour.material.depth_write = False
            else:
                contour.material.alpha_mode = "auto"
                contour.material.depth_write = True

    def toggle_roi_visibility_at(self, index):
        """Toggle visibility for a single ROI actor.

        Parameters
        ----------
        index : int
            Index of the ROI actor whose visibility should flip.
        """
        if index < 0 or index >= len(self._visualizations["roi"]):
            return
        actor = self._visualizations["roi"][index]
        actor.visible = not actor.visible
        path = input_manager.provided_roi_paths[index]
        self._roi_visibility[path] = bool(actor.visible)

    def is_roi_visible_at(self, index):
        """Return whether the ROI actor at ``index`` is currently shown."""
        if index < 0 or index >= len(self._visualizations["roi"]):
            return True
        return bool(self._visualizations["roi"][index].visible)

    def toggle_roi_applied_at(self, index):
        """Toggle whether the ROI at ``index`` participates in the filter.

        Parameters
        ----------
        index : int
            Index into the provided ROI list whose applied flag is flipped.
        """
        if index < 0 or index >= len(input_manager.provided_roi_paths):
            return
        path = input_manager.provided_roi_paths[index]
        self._roi_applied[path] = not self._roi_applied.get(path, True)

    def is_roi_applied_at(self, index):
        """Return whether the ROI at ``index`` contributes to the filter.

        Parameters
        ----------
        index : int
            Index into the provided ROI list.

        Returns
        -------
        bool
            True if the ROI is applied (default), False otherwise.
        """
        if index < 0 or index >= len(input_manager.provided_roi_paths):
            return True
        path = input_manager.provided_roi_paths[index]
        return bool(self._roi_applied.get(path, True))

    def toggle_roi_negated_at(self, index):
        """Toggle whether the ROI at ``index`` is negated in the filter.

        Parameters
        ----------
        index : int
            Index into the provided ROI list whose negation flag is flipped.
        """
        if index < 0 or index >= len(input_manager.provided_roi_paths):
            return
        path = input_manager.provided_roi_paths[index]
        self._roi_negated[path] = not self._roi_negated.get(path, False)

    def is_roi_negated_at(self, index):
        """Return whether the ROI at ``index`` is negated in the filter.

        Parameters
        ----------
        index : int
            Index into the provided ROI list.

        Returns
        -------
        bool
            True if the ROI mask is inverted before AND-combining, else False.
        """
        if index < 0 or index >= len(input_manager.provided_roi_paths):
            return False
        path = input_manager.provided_roi_paths[index]
        return bool(self._roi_negated.get(path, False))

    def set_roi_opacity(self, value):
        """Set opacity for every ROI actor from a 0–100 slider value.

        Parameters
        ----------
        value : int
            The slider value (0-100).
        """
        state_manager.roi_opacity = value
        opacity = value / 100.0
        for roi_actor in self._visualizations["roi"]:
            self._apply_roi_opacity(roi_actor, opacity)

    def get_roi_color(self, index):
        """Return the assigned RGB color of an ROI actor.

        Parameters
        ----------
        index : int
            Index of the ROI actor.

        Returns
        -------
        tuple of float or None
            The RGB color, or None if ``index`` is out of range.
        """
        if index < 0 or index >= len(self._visualizations["roi"]):
            return None
        return getattr(self._visualizations["roi"][index], "color", None)

    @property
    def roi_visualizations(self):
        """Return the list of ROI actors."""
        return self._visualizations["roi"]

    def apply_roi_filter(self):
        """Filter the tractogram streamlines using positive and negated ROIs.

        Positive ROIs (the default) are AND-combined and the streamlines
        that pass through the resulting mask are kept. Negated ROIs are
        OR-combined and any streamline touching that combined mask is then
        subtracted from the kept set. When every applied ROI is negated,
        the kept set starts as all streamlines before the subtraction.

        The cluster state is invalidated so the next call to
        :meth:`_apply_tractogram_states` re-clusters on the filtered set.
        Clears the filter when no ROI is loaded or none is applied.

        The reference shape and affine come from the T1 image when one is
        loaded (matching the legacy ``app.py`` pipeline) and otherwise
        fall back to the first applied ROI's own shape and affine.

        Returns
        -------
        bool
            True when the latest state was modified and the tractogram
            visualization needs rebuilding, False otherwise.
        """
        if not state_manager.has_states():
            return False
        latest_state = state_manager.get_latest_state()

        if not input_manager.has_roi:
            if latest_state.filtered_streamline_ids is None:
                return False
            latest_state.filtered_streamline_ids = None
            latest_state.tractogram_states = None
            return True

        if not input_manager.has_tractogram:
            return False

        sft, _, _, _ = input_manager.get_current_tractogram()

        positive_volumes = []
        negative_volumes = []
        first_applied_index = None
        for index in range(len(input_manager.provided_roi_paths)):
            if not self.is_roi_applied_at(index):
                continue
            roi_volume, _roi_affine, _, _ = input_manager.get_roi_at(index)
            if first_applied_index is None:
                first_applied_index = index
            if self.is_roi_negated_at(index):
                negative_volumes.append(roi_volume)
            else:
                positive_volumes.append(roi_volume)

        if first_applied_index is None:
            if latest_state.filtered_streamline_ids is None:
                return False
            latest_state.filtered_streamline_ids = None
            latest_state.tractogram_states = None
            return True

        if input_manager.has_t1:
            t1_volume, reference_affine, _, _ = input_manager.get_current_t1()
            reference_shape = t1_volume.shape
        else:
            _, reference_affine, _, _ = input_manager.get_roi_at(first_applied_index)
            reference_shape = (positive_volumes or negative_volumes)[0].shape

        if positive_volumes:
            try:
                positive_mask = calculate_filter(
                    positive_volumes, reference_shape=reference_shape
                )
            except ValueError:
                return False
            world_mask, origin = transform_roi_to_world_grid(
                positive_mask, reference_affine
            )
            kept_ids = {
                int(i)
                for i in filter_streamline_ids(
                    sft.streamlines, world_mask, origin=origin
                )
            }
        else:
            kept_ids = set(range(len(sft.streamlines)))

        if negative_volumes:
            negative_mask = np.zeros(reference_shape, dtype=bool)
            matched = 0
            for volume in negative_volumes:
                volume_mask = np.asarray(volume).astype(bool, copy=False)
                if volume_mask.shape != reference_shape:
                    continue
                negative_mask |= volume_mask
                matched += 1
            if matched:
                world_mask, origin = transform_roi_to_world_grid(
                    negative_mask, reference_affine
                )
                excluded_ids = {
                    int(i)
                    for i in filter_streamline_ids(
                        sft.streamlines, world_mask, origin=origin
                    )
                }
                kept_ids -= excluded_ids

        latest_state.filtered_streamline_ids = np.asarray(
            sorted(kept_ids), dtype=np.int32
        )
        latest_state.tractogram_states = None
        return True


visualization_manager = VisualizationManager()
