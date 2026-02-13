import logging

from PySide6.QtWidgets import QApplication, QMainWindow
import numpy as np

from fury import window
from fury.actor import Group, set_group_visibility, show_slices
from fury.colormap import distinguishable_colormap
from fury.lib import (
    DirectionalLight,
    Event,
    OrthographicCamera,
    PanZoomController,
    PerspectiveCamera,
    TrackballController,
)
from tractome.compute import (
    calculate_filter,
    filter_streamline_ids,
    mkbm_clustering,
    transform_roi_to_world_grid,
)
from tractome.io import (
    read_mesh,
    read_nifti,
    read_tractogram,
    save_tractogram_from_streamlines,
)
from tractome.mem import ClusterState, StateManager
from tractome.ui import (
    STYLE_SHEET,
    create_cluster_selection_buttons,
    create_clusters_slider,
    create_mesh_controls,
    create_roi_controls,
    create_slice_sliders,
    create_ui,
    update_history_table,
)
from tractome.viz import (
    _deselect_streamtube,
    _select_streamtube,
    _toggle_streamtube_selection,
    create_image_slicer,
    create_keystroke_card,
    create_mesh,
    create_roi,
    create_streamlines,
    create_streamlines_projection,
    create_streamtube,
)

app = QApplication([])
# Qt.Checked


class Tractome(QMainWindow):
    def __init__(
        self, tractogram=None, mesh=None, mesh_texture=None, t1=None, roi=None
    ):
        """Initialize the Tractome application.

        Parameters
        ----------
        tractogram : str, optional
            The file path to the tractogram
        mesh : str, optional
            The file path to the mesh
        mesh_texture : str, optional
            The file path to the mesh texture,
        t1 : str, optional
            The file path to the T1 image
        roi : str or Sequence[str], optional
            One or more file paths to ROI files.
        """
        super().__init__()
        self.tractogram = tractogram
        self.mesh = mesh
        self.mesh_texture = mesh_texture
        self.t1 = t1
        if roi is None:
            self.rois = []
        elif isinstance(roi, (list, tuple)):
            self.rois = list(roi)
        else:
            self.rois = [roi]
        self._mode = "3D"
        self._3D_actors = {"t1": None, "tractogram": None, "mesh": None, "roi": None}
        self._2D_actors = {"t1": None, "tractogram": None, "mesh": None, "roi": []}
        self._sft = None
        self._clusters = None
        self._cluster_reps = {}
        self._streamline_bundles = []
        self._selected_clusters = set()
        self._streamline_projections = []
        self._streamline_bounds_min = None
        self._streamline_bounds_max = None
        self._roi_actors = []
        self._roi_slice_actors = []
        self._roi_controls_widget = None
        self._roi_checkboxes = []
        self._roi_filter = None
        self._roi_filtered_ids = None
        self._roi_origin = (0, 0, 0)
        self._affine = None
        self._bounds = None
        self._mesh_mode = "Normals"
        self._state_manager = StateManager()
        self._focused_actor = None
        self._init_UI()
        self._init_actors()

    def _build_roi_rgba_volume(self, volume, color):
        """Create an RGBA volume for a single ROI with transparent background."""
        rgba = np.zeros((*volume.shape, 3), dtype=np.float32)
        mask = volume != 0
        if np.any(mask):
            rgba[mask, :3] = color
        return rgba

    def _init_UI(self):
        """Initialize the user interface."""
        self.resize(800, 800)
        self.setWindowTitle("Tractome 2.0")

        self._3D_scene = window.Scene(background=(0.2, 0.2, 0.2))
        self._2D_scene = window.Scene(background=(0.2, 0.2, 0.2))

        self._3D_scene.add()
        self._2D_scene.add(DirectionalLight())

        self._3D_camera = PerspectiveCamera()

        self._3D_camera.add(DirectionalLight())
        self._3D_scene.add(self._3D_camera)
        self._2D_camera = OrthographicCamera()

        self.show_manager = window.ShowManager(
            scene=self._3D_scene,
            camera=self._3D_camera,
            qt_app=app,
            qt_parent=self,
            window_type="qt",
        )

        # TODO: Remove long press event handler for Qt
        # This is a temporary workaround for the long press issue in Qt
        self.show_manager.renderer.remove_event_handler(
            self.show_manager._set_key_long_press_event, "key_up", "key_down"
        )

        def _register_clicks(event):
            """Handle selection clicks.

            Parameters
            ----------
            event : Event
                The click event.
            """
            if event.type == "pointer_down":
                self._focused_actor = event.target
            elif event.type == "pointer_up" and self._focused_actor != event.target:
                self._focused_actor = None
            elif event.type == "pointer_up" and self._focused_actor == event.target:
                event = Event(
                    type="on_selection", target=self._focused_actor, bubbles=False
                )
                self.show_manager.renderer.dispatch_event(event)
                self._focused_actor = None
            else:
                self._focused_actor = None

        self.show_manager.renderer.add_event_handler(
            _register_clicks, "pointer_down", "pointer_up"
        )

        self._3D_controller = TrackballController(
            self._3D_camera, register_events=self.show_manager.renderer
        )
        self._2D_controller = PanZoomController(
            self._2D_camera, register_events=self.show_manager.renderer
        )
        self._2D_controller.enabled = False

        (
            main_widget,
            self.left_panel,
            self.right_panel,
            self._toggle_3d,
            self._toggle_2d,
            self._reset_view,
            self._toggle_suggestion,
        ) = create_ui(self.show_manager.window)
        self._toggle_3d.clicked.connect(self.toggle_3D_mode)
        self._toggle_2d.clicked.connect(self.toggle_2D_mode)
        self._reset_view.clicked.connect(self.reset_view)
        self._toggle_suggestion.clicked.connect(self.toggle_suggestion)
        self.setCentralWidget(main_widget)
        self.setStyleSheet(STYLE_SHEET)

        # Toggle Control Widgets
        self._slice_widget = None
        self._3D_check_box_values = None
        self._2D_radio_buttons_values = None
        self._colors_gen = distinguishable_colormap()

    def _init_actors(self):
        """Initialize the actors for the scene."""
        if self.tractogram:
            self._sft = read_tractogram(self.tractogram, reference=self.t1)
            self._precompute_streamline_bounds()
            if (
                self._sft.data_per_streamline is None
                or "dismatrix" not in self._sft.data_per_streamline
            ):
                color = next(self._colors_gen)
                tractogram = create_streamlines(self._sft.streamlines, color=color)
                self._3D_scene.add(tractogram)
                self._3D_actors["tractogram"] = tractogram
            else:
                self._state_manager.add_state(
                    ClusterState(100, np.arange(len(self._sft.streamlines)), 1000)
                )
                self.perform_clustering(value=100)
                (
                    self._cluster_widget,
                    self._cluster_input,
                    self._apply_button,
                    self._prev_state_button,
                    self._next_state_button,
                    self._history_table,
                ) = create_clusters_slider(default_value=100)
                self.left_panel.layout().addWidget(self._cluster_widget)
                self._apply_button.clicked.connect(self.on_apply_clusters)
                self._cluster_input.lineEdit().returnPressed.connect(
                    self.on_apply_clusters
                )
                self._prev_state_button.clicked.connect(self.on_prev_state)
                self._next_state_button.clicked.connect(self.on_next_state)
                self._update_history_table()

                (
                    self._cluster_selection_widget,
                    self._select_all_button,
                    self._select_none_button,
                    self._swap_selection_button,
                    self._delete_selected_button,
                    self._expand_button,
                    self._collapse_button,
                    self._show_button,
                    self._hide_button,
                ) = create_cluster_selection_buttons()
                self.left_panel.layout().addWidget(self._cluster_selection_widget)
                self._select_all_button.clicked.connect(self.on_select_all)
                self._select_none_button.clicked.connect(self.on_select_null)
                self._swap_selection_button.clicked.connect(self.on_swap_selection)
                self._delete_selected_button.clicked.connect(self.delete_selection)
                self._expand_button.clicked.connect(self.on_expand_clusters)
                self._collapse_button.clicked.connect(self.collapse_streamline_bundles)
                self._show_button.clicked.connect(self.on_show_clusters)
                self._hide_button.clicked.connect(self.on_hide_clusters)

                self._keystroke_card = create_keystroke_card()
                self._3D_scene.ui_scene.add(self._keystroke_card)

            self.show_manager.renderer.add_event_handler(
                self.handle_key_strokes, "key_down"
            )

        if self.t1:
            nifti_img, affine = read_nifti(self.t1)
            self._bounds = nifti_img.shape
            self._affine = affine
            image_slicer = create_image_slicer(nifti_img, affine=affine)
            self._3D_scene.add(image_slicer)

            image_slice = create_image_slicer(
                nifti_img, affine=affine, mode="weighted_blend", depth_write=False
            )
            set_group_visibility(image_slice, (False, False, True))
            self._2D_scene.add(image_slice)

            min_vals, max_vals = image_slicer.get_bounding_box()
            self._slice_widget, sliders, checkboxes, _ = create_slice_sliders(
                min_vals=np.asarray(min_vals, dtype=np.int32),
                max_vals=np.asarray(max_vals, dtype=np.int32),
            )
            self.left_panel.layout().addWidget(self._slice_widget)
            self._x_slider, self._y_slider, self._z_slider = sliders

            for slider in sliders:
                slider.valueChanged.connect(self.update_slices)

            self._x_checkbox, self._y_checkbox, self._z_checkbox = checkboxes

            for checkbox in checkboxes:
                checkbox.stateChanged.connect(self.update_slice_visibility)

            self._3D_actors["t1"] = image_slicer
            self._2D_actors["t1"] = image_slice

        if self.mesh:
            self._create_mesh_actor()
            (
                self._mesh_controls_widget,
                self._mesh_opacity_slider,
                self._mesh_visibility_checkbox,
                self._mesh_mode_group,
                self._normals_radio,
                self._photographic_radio,
            ) = create_mesh_controls()
            self.left_panel.layout().addWidget(self._mesh_controls_widget)
            self._mesh_visibility_checkbox.stateChanged.connect(
                self.toggle_mesh_visibility
            )
            self._mesh_opacity_slider.valueChanged.connect(self.update_mesh_opacity)
            self._mesh_mode_group.buttonClicked.connect(self.on_mesh_mode_changed)

        for roi_path in self.rois:
            roi_nifti, affine = read_nifti(roi_path)

            # Fail-safe if a t1 image is not provided.
            # Use the first ROI's shape and affine as the reference for all subsequent
            # ROIs and filtering.
            if self._bounds is None:
                self._bounds = roi_nifti.shape
            if self._affine is None:
                self._affine = affine

            color = next(self._colors_gen)
            roi_object = create_roi(roi_nifti, affine=affine, color=color)
            self._3D_scene.add(roi_object)
            self._roi_actors.append(roi_object)

            roi_rgba = self._build_roi_rgba_volume(roi_nifti, color)
            roi_slice = create_image_slicer(
                roi_rgba, affine=affine, mode="weighted_blend", depth_write=False
            )
            for slice_actor in roi_slice.children:
                slice_actor.material.opacity = 0.3
            set_group_visibility(roi_slice, (False, False, True))
            self._2D_scene.add(roi_slice)
            self._roi_slice_actors.append(roi_slice)
            self.update_slices(None)

        if self._roi_slice_actors:
            self._2D_actors["roi"] = self._roi_slice_actors

        if self._roi_actors:
            (
                self._roi_controls_widget,
                self._roi_checkboxes,
            ) = create_roi_controls(self.rois)
            self.left_panel.layout().addWidget(self._roi_controls_widget)
            for checkbox, roi_actor, roi_slice in zip(
                self._roi_checkboxes, self._roi_actors, self._roi_slice_actors
            ):
                checkbox.stateChanged.connect(
                    lambda state, actor=roi_actor, slice_actor=roi_slice: (
                        self.toggle_roi_visibility(actor, state, slice_actor)
                    )
                )

            if self.tractogram:
                self._apply_roi_filter()
                self.perform_clustering(value=100)
        self.show_manager.start()

    def _create_roi_filter(self):
        rois = []
        for idx, roi_path in enumerate(self.rois):
            if self._roi_checkboxes[idx].isChecked():
                roi_nifti, _ = read_nifti(roi_path)
                rois.append(roi_nifti)
        if rois:
            self._roi_filter = calculate_filter(rois, reference_shape=self._bounds)
            self._roi_filter, self._roi_origin = transform_roi_to_world_grid(
                self._roi_filter, self._affine
            )
        else:
            self._roi_filter = None

    def _apply_roi_filter(self):
        self._create_roi_filter()
        self._roi_filtered_ids = filter_streamline_ids(
            self._sft.streamlines, self._roi_filter, origin=self._roi_origin
        )

    def _create_mesh_actor(self, mode="Normals"):
        """Create a 3D mesh actor from the loaded mesh."""
        mesh_obj, texture = read_mesh(self.mesh, texture=self.mesh_texture)
        mesh_actor = create_mesh(mesh_obj, texture=texture, mode=mode)
        self._3D_scene.add(mesh_actor)
        self._3D_actors["mesh"] = mesh_actor

    def _precompute_streamline_bounds(self):
        """Cache min/max xyz bounds per streamline for fast selection queries."""
        if self._sft is None:
            self._streamline_bounds_min = None
            self._streamline_bounds_max = None
            return

        n_streamlines = len(self._sft.streamlines)
        mins = np.zeros((n_streamlines, 3), dtype=np.float32)
        maxs = np.zeros((n_streamlines, 3), dtype=np.float32)

        for idx, line in enumerate(self._sft.streamlines):
            pts = np.asarray(line, dtype=np.float32)
            mins[idx] = pts.min(axis=0)
            maxs[idx] = pts.max(axis=0)

        self._streamline_bounds_min = mins
        self._streamline_bounds_max = maxs

    def _selected_streamline_ids(self):
        """Return unique streamline ids for currently selected clusters."""
        if not self._selected_clusters or not self._clusters:
            return np.asarray([], dtype=np.int32)

        ids = []
        for cluster in self._selected_clusters:
            ids.extend(self._clusters.get(cluster.rep, []))
        if not ids:
            return np.asarray([], dtype=np.int32)

        return np.unique(np.asarray(ids, dtype=np.int32))

    def _clamp_slice_values_to_selection(self, slice_values):
        """Keep 2D slice values inside selected-streamline bounds when possible."""
        if slice_values is None:
            return None
        if self._streamline_bounds_min is None or self._streamline_bounds_max is None:
            return slice_values

        selected_ids = self._selected_streamline_ids()
        if selected_ids.size == 0:
            return slice_values

        mins = self._streamline_bounds_min[selected_ids].min(axis=0)
        maxs = self._streamline_bounds_max[selected_ids].max(axis=0)
        sliders = (self._x_slider, self._y_slider, self._z_slider)
        clamped = [int(v) for v in slice_values]
        changed = False

        for axis, (slider, cur_val) in enumerate(zip(sliders, clamped)):
            lower = max(float(slider.minimum()), float(mins[axis]))
            upper = min(float(slider.maximum()), float(maxs[axis]))
            if lower > upper:
                continue
            if cur_val < lower or cur_val > upper:
                clamped_val = int(np.rint((lower + upper) / 2.0))
                clamped[axis] = clamped_val
                changed = True

        if changed:
            for slider, val in zip(sliders, clamped):
                slider.blockSignals(True)
                slider.setValue(int(val))
                slider.blockSignals(False)

        return tuple(clamped)

    def on_mesh_mode_changed(self, button):
        """Handle mesh mode radio button change."""
        mode = button.text()
        if mode.lower() != self._mesh_mode.lower():
            self._3D_scene.remove(self._3D_actors["mesh"])
            self._mesh_mode = mode
            self._create_mesh_actor(mode=mode)
            self.show_manager.render()

    def get_current_slider_position(self):
        """Get the current position of the slice sliders.

        Returns
        -------
        tuple
            A tuple containing the current values of the X, Y, and Z sliders.
        """
        if hasattr(self, "_x_slider"):
            return (
                self._x_slider.value(),
                self._y_slider.value(),
                self._z_slider.value(),
            )
        return None

    def get_current_checkbox_states(self):
        """Get the current state of the slice checkboxes.

        Returns
        -------
        tuple
            A tuple containing the current checked states of the X, Y, and Z checkboxes.
            True means checked, False means unchecked.
        """
        if hasattr(self, "_x_checkbox"):
            return (
                self._x_checkbox.isChecked(),
                self._y_checkbox.isChecked(),
                self._z_checkbox.isChecked(),
            )
        return None

    def update_slices(self, _value):
        """Update the visible slices of the 3D actors. Callback for slider changes.

        Parameters
        ----------
        _value : int
            The current value of the slider.
        """
        slices = self.get_current_slider_position()
        show_slices(self._3D_actors["t1"], slices)
        show_slices(self._2D_actors["t1"], slices)
        for roi_slice in self._roi_slice_actors:
            show_slices(roi_slice, np.asarray(slices) + 0.1)
        for projection in self._streamline_projections:
            show_slices(projection, slices)
        self.show_manager.render()

    def _update_history_table(self):
        """Update the history table with the latest data."""
        all_states = self._state_manager.get_all_states()
        current_index = self._state_manager.get_current_index()
        update_history_table(self._history_table, all_states, current_index)

    def update_slice_visibility(self, _value):
        """Update the visibility of the slices based on checkbox states. Callback
        for the checkbox changes.

        Parameters
        ----------
        _value : int
            The current checked state of the checkbox.
        """
        if self._mode == "3D":
            checkbox_states = self.get_current_checkbox_states()
            set_group_visibility(self._3D_actors["t1"], checkbox_states)
        elif self._mode == "2D":
            radio_states = self.get_current_checkbox_states()
            self._2D_camera.show_object(
                self._2D_actors["t1"],
                tuple(-1 * np.asarray(radio_states, dtype=int)),
            )
            set_group_visibility(self._2D_actors["t1"], radio_states)
            for roi_slice in self._roi_slice_actors:
                set_group_visibility(roi_slice, radio_states)
            for proj in self._streamline_projections:
                set_group_visibility(proj, radio_states)
        self.show_manager.render()

    def toggle_mesh_visibility(self, state):
        """Toggle the visibility of the mesh in the 3D scene.

        Parameters
        ----------
        state : int
            The checked state of the checkbox (Qt.Checked or Qt.Unchecked).
        """
        self._toggle_visibility(self._3D_actors["mesh"], state)

    def toggle_roi_visibility(self, roi, state, roi_slice=None):
        """Toggle visibility of a single ROI actor (and its 2D slice overlay).

        Parameters
        ----------
        roi : Actor
            The ROI actor to toggle visibility for.
        state : int
            The checked state of the checkbox (Qt.Checked or Qt.Unchecked).
        roi_slice : Actor, optional
            The corresponding 2D slice actor, if available.
        """
        self._toggle_visibility(roi, state)

        CHECKED = 2  # Qt.Checked
        if roi_slice is not None:
            if state == CHECKED:
                if roi_slice not in self._2D_scene.main_scene.children:
                    self._2D_scene.add(roi_slice)
                axes_states = self.get_current_checkbox_states()
                if axes_states is None:
                    axes_states = (True, True, True)
                set_group_visibility(roi_slice, axes_states)
            else:
                if roi_slice in self._2D_scene.main_scene.children:
                    self._2D_scene.remove(roi_slice)

        self._apply_roi_filter()
        self.perform_clustering()
        self.show_manager.render()

    def _toggle_visibility(self, actor, state):
        """Toggle the visibility of an actor in the 3D scene.

        Parameters
        ----------
        actor : Actor
            The actor to toggle visibility for.
        state : int
            The checked state of the checkbox (Qt.Checked or Qt.Unchecked).
        """
        CHECKED = 2  # Qt.Checked
        if state == CHECKED:
            actor.visible = True
        else:
            actor.visible = False

    def update_mesh_opacity(self, value):
        """Update the opacity of the mesh.

        Parameters
        ----------
        value : int
            The slider value (0-100).
        """
        opacity = value / 100.0
        self._3D_actors["mesh"].material.opacity = opacity
        if opacity < 1.0:
            self._3D_actors["mesh"].material.alpha_mode = "blend"
            self._3D_actors["mesh"].material.depth_write = False
        else:
            self._3D_actors["mesh"].material.alpha_mode = "solid"
            self._3D_actors["mesh"].material.depth_write = True
        self.show_manager.render()

    def perform_clustering(self, *, value=None):
        """Perform clustering on the current data.

        Parameters
        ----------
        value : int
            The number of clusters to create.
        """

        if not self._state_manager.has_states():
            logging.info("No states available for clustering.")
            return

        latest_state = self._state_manager.get_latest_state()

        if value is None:
            value = latest_state.nb_clusters

        streamline_ids = latest_state.streamline_ids
        self._selected_clusters.clear()
        self.collapse_streamline_bundles()
        if self._roi_filtered_ids:
            streamline_ids = list(
                set(streamline_ids).intersection(set(self._roi_filtered_ids))
            )
        self._clusters = mkbm_clustering(
            self._sft.data_per_streamline["dismatrix"],
            n_clusters=value,
            streamline_ids=streamline_ids,
        )

        for cluster in self._cluster_reps.values():
            if cluster in self._3D_scene.main_scene.children:
                self._3D_scene.remove(cluster)

        for bundle in self._streamline_bundles:
            if bundle in self._3D_scene.main_scene.children:
                self._3D_scene.remove(bundle)

        self._cluster_reps = create_streamtube(self._clusters, self._sft.streamlines)
        for cluster in self._cluster_reps.values():
            cluster.add_event_handler(self.toggle_cluster_selection, "on_selection")
            self._3D_scene.add(cluster)
        self.show_manager.render()
        self._last_clustered_value = value
        if hasattr(self, "_cluster_input"):
            self._cluster_input.setValue(value)

    def on_next_state(self):
        """Handle the 'Next State' button click."""
        if self._state_manager.can_move_next():
            latest_state = self._state_manager.move_next()
            self._cluster_input.setMaximum(latest_state.max_clusters)
            self._cluster_input.setValue(latest_state.nb_clusters)
            self.perform_clustering(value=latest_state.nb_clusters)
            self._update_history_table()
        else:
            logging.warning("No next state available.")

    def on_prev_state(self):
        """Handle the 'Previous State' button click."""
        if self._state_manager.can_move_back():
            latest_state = self._state_manager.move_back()
            self._cluster_input.setMaximum(latest_state.max_clusters)
            self._cluster_input.setValue(latest_state.nb_clusters)
            self.perform_clustering(value=latest_state.nb_clusters)
            self._update_history_table()
        else:
            logging.warning("No previous state available.")

    def delete_selection(self):
        """Delete the selected clusters from the current state."""
        streamline_ids = []
        for cluster in self._selected_clusters:
            streamline_ids.extend(self._clusters[cluster.rep])

        if len(streamline_ids) > 0:
            for cluster in self._cluster_reps.values():
                if (
                    cluster in self._3D_scene.main_scene.children
                    and cluster not in self._selected_clusters
                ):
                    self._3D_scene.remove(cluster)
            self._cluster_reps.clear()
            for cluster in self._selected_clusters:
                self._cluster_reps[cluster.rep] = cluster
            old_max = self._cluster_input.maximum()
            old_value = (
                self._cluster_input.value() if hasattr(self, "_cluster_input") else 100
            )
            new_max = min(1000, len(streamline_ids))

            self._cluster_input.setMaximum(new_max)

            new_value = len(self._selected_clusters)
            if old_max > new_max:
                new_value = int((old_value / old_max) * new_max)

            self._cluster_input.setValue(new_value)
            self._state_manager.add_state(
                ClusterState(
                    new_value,
                    np.array(streamline_ids),
                    self._cluster_input.maximum(),
                )
            )

            self._update_history_table()
            self.show_manager.render()
        else:
            logging.warning("No clusters selected to save a state.")

    def on_expand_clusters(self):
        """Expand all selected clusters into streamline bundles."""
        for cluster in self._selected_clusters:
            if cluster in self._3D_scene.main_scene.children:
                self._3D_scene.remove(cluster)
            streamlines = [
                np.asarray(self._sft.streamlines[line])
                for line in self._clusters[cluster.rep]
            ]
            streamlines = create_streamlines(
                streamlines,
                cluster.geometry.colors.data[0],
            )
            streamlines.rep = cluster.rep
            self._streamline_bundles.append(streamlines)
            self._3D_scene.add(streamlines)

    def on_show_clusters(self):
        """Show all clusters in the 3D scene."""
        for cluster in self._cluster_reps.values():
            if (
                cluster not in self._3D_scene.main_scene.children
                and cluster not in self._selected_clusters
            ):
                self._3D_scene.add(cluster)

    def on_hide_clusters(self):
        """Hide all unselected clusters from the 3D scene."""
        for cluster in self._cluster_reps.values():
            if (
                cluster not in self._selected_clusters
                and cluster in self._3D_scene.main_scene.children
            ):
                self._3D_scene.remove(cluster)

    def on_swap_selection(self):
        """Handle the 'Swap Selection' button click."""
        for cluster in self._cluster_reps.values():
            self._toggle_cluster_selection(cluster)
            _toggle_streamtube_selection(cluster)
        self.show_manager.render()

    def on_select_null(self):
        """Handle the 'Select Null' button click."""
        self._selected_clusters.clear()
        for cluster in self._cluster_reps.values():
            _deselect_streamtube(cluster)
        self.show_manager.render()

    def on_select_all(self):
        """Handle the 'Select All' button click."""
        self._selected_clusters = set(self._cluster_reps.values())
        for cluster in self._cluster_reps.values():
            _select_streamtube(cluster)
        self.show_manager.render()

    def _toggle_cluster_selection(self, cluster):
        """Toggle the selection state of a cluster.

        Parameters
        ----------
        cluster : Actor
            The cluster actor to toggle.
        """
        if cluster in self._selected_clusters:
            self._selected_clusters.remove(cluster)
        else:
            self._selected_clusters.add(cluster)

    def toggle_cluster_selection(self, event):
        """Toggle the selection state of a cluster.

        Parameters
        ----------
        event : Event
            The click event.
        """
        cluster = event.target
        self._toggle_cluster_selection(cluster)
        self.show_manager.render()

    def handle_key_strokes(self, event):
        if event.key == "e":
            self.on_expand_clusters()
        elif event.key == "c":
            self.collapse_streamline_bundles()
        elif event.key == "h":
            self.on_hide_clusters()
        elif event.key == "s":
            self.on_show_clusters()
        elif event.key == "a":
            self.on_select_all()
        elif event.key == "n":
            self.on_select_null()
        elif event.key == "i":
            self.on_swap_selection()
        elif event.key == "d":
            self.delete_selection()
        elif event.key == "r":
            self.reset_view()
        elif event.key == "x":
            self.toggle_suggestion()
        elif event.key == "t":
            latest_state = self._state_manager.get_latest_state()
            streamline_ids = np.asarray(latest_state.streamline_ids, dtype=np.int32)
            save_tractogram_from_streamlines(
                self._sft.streamlines[streamline_ids],
                self.t1,
                self._sft.data_per_streamline["dismatrix"][streamline_ids],
                file_path=(
                    f"state_{self._state_manager.get_current_index()}"
                    f"_{len(latest_state.streamline_ids.ravel())}.trk"
                ),
            )

        self.show_manager.render()

    def collapse_streamline_bundles(self):
        """Collapse all streamline bundles."""
        for bundle in self._streamline_bundles:
            self._3D_scene.remove(bundle)
            self._3D_scene.add(self._cluster_reps[bundle.rep])
        self._streamline_bundles = []

    def _create_streamlines_projection(self):
        if self._streamline_projections:
            self._2D_scene.remove(*self._streamline_projections)
        self._streamline_projections.clear()
        slice_values = self._clamp_slice_values_to_selection(
            self.get_current_slider_position()
        )
        for cluster in self._selected_clusters:
            streamlines = [
                np.asarray(self._sft.streamlines[line])
                for line in self._clusters[cluster.rep]
            ]
            projection = create_streamlines_projection(
                streamlines=streamlines,
                colors=cluster.geometry.colors.data[0],
                slice_values=slice_values,
            )
            self._streamline_projections.append(projection)
        if self._streamline_projections:
            self._2D_scene.add(*self._streamline_projections)
        self._2D_actors["tractogram"] = self._streamline_projections

    def toggle_3D_mode(self):
        """Toggle to 3D mode."""
        if self._mode != "3D":
            if self.tractogram and self._cluster_widget:
                self._cluster_widget.show()
                self._cluster_selection_widget.show()
            self._mode = "3D"
            self.show_manager.screens[0].scene = self._3D_scene
            self.show_manager.screens[0].camera = self._3D_camera
            self.show_manager.screens[0].controller = self._3D_controller
            self._3D_controller.enabled = True
            self._2D_controller.enabled = False

            if self._2D_radio_buttons_values is not None:
                self._3D_camera.show_object(
                    self._3D_actors["t1"],
                    tuple(-1 * np.asarray(self._2D_radio_buttons_values, dtype=int)),
                )
            self.show_manager.render()

            # Safely remove and delete the existing widget
            if self._slice_widget is not None:
                self.left_panel.layout().removeWidget(self._slice_widget)
                self._slice_widget.deleteLater()

            min_vals, max_vals = self._3D_actors["t1"].get_bounding_box()
            self._slice_widget, sliders, checkboxes, _ = create_slice_sliders(
                min_vals=np.asarray(min_vals, dtype=np.int32),
                max_vals=np.asarray(max_vals, dtype=np.int32),
                control_type="checkbox",
                default_vals=self.get_current_slider_position(),
            )
            self.left_panel.layout().addWidget(self._slice_widget)
            self._x_slider, self._y_slider, self._z_slider = sliders

            for slider in sliders:
                slider.valueChanged.connect(self.update_slices)

            self._2D_radio_buttons_values = self.get_current_checkbox_states()

            if self._3D_check_box_values is None:
                self._3D_check_box_values = (True, True, True)

            self._x_checkbox, self._y_checkbox, self._z_checkbox = checkboxes
            for checkbox, value in zip(checkboxes, self._3D_check_box_values):
                checkbox.setChecked(value)
                checkbox.stateChanged.connect(self.update_slice_visibility)

            if hasattr(self, "_mesh_controls_widget"):
                self._mesh_controls_widget.show()
            if self._roi_controls_widget is not None:
                self._roi_controls_widget.show()
            if hasattr(self, "_reset_view"):
                self._reset_view.show()
            if hasattr(self, "_toggle_suggestion"):
                self._toggle_suggestion.show()

    def toggle_2D_mode(self):
        """Toggle to 2D mode."""
        if self._mode != "2D":
            if self.t1 is None:
                logging.warning("2D mode requires a T1 image to be loaded.")
                return
            if self.tractogram and self._cluster_widget:
                self._cluster_widget.hide()
                self._cluster_selection_widget.hide()
            self._mode = "2D"
            self.show_manager.screens[0].scene = self._2D_scene
            self.show_manager.screens[0].camera = self._2D_camera
            self.show_manager.screens[0].controller = self._2D_controller
            self._3D_controller.enabled = False
            self._2D_controller.enabled = True
            self._create_streamlines_projection()

            if hasattr(self, "_mesh_controls_widget"):
                self._mesh_controls_widget.hide()
            if self._roi_controls_widget is not None:
                self._roi_controls_widget.hide()
            if hasattr(self, "_reset_view"):
                self._reset_view.hide()
            if hasattr(self, "_toggle_suggestion"):
                self._toggle_suggestion.hide()

            # Safely remove and delete the existing widget
            if self._slice_widget is not None:
                self.left_panel.layout().removeWidget(self._slice_widget)
                self._slice_widget.deleteLater()

            min_vals, max_vals = self._3D_actors["t1"].get_bounding_box()
            self._slice_widget, sliders, radio_buttons, button_group = (
                create_slice_sliders(
                    min_vals=np.asarray(min_vals, dtype=np.int32),
                    max_vals=np.asarray(max_vals, dtype=np.int32),
                    control_type="radio",
                    default_vals=self.get_current_slider_position(),
                )
            )
            self.left_panel.layout().addWidget(self._slice_widget)
            self._x_slider, self._y_slider, self._z_slider = sliders

            for slider in sliders:
                slider.valueChanged.connect(self.update_slices)

            self._3D_check_box_values = self.get_current_checkbox_states()

            if self._2D_radio_buttons_values is None:
                self._2D_radio_buttons_values = (False, False, True)

            self._x_checkbox, self._y_checkbox, self._z_checkbox = radio_buttons

            for radio_button, value in zip(
                radio_buttons, self._2D_radio_buttons_values
            ):
                radio_button.setChecked(value)
                radio_button.clicked.connect(self.update_slice_visibility)

            radio_states = self.get_current_checkbox_states()
            self._2D_camera.show_object(
                self._2D_actors["t1"], tuple(-1 * np.asarray(radio_states, dtype=int))
            )
            self.update_slice_visibility(None)
            self.show_manager.render()

    def on_apply_clusters(self):
        """Handle the apply clusters button or editing finished."""
        value = self._cluster_input.value()
        if 1 <= value <= self._cluster_input.maximum():
            self.perform_clustering(value)
        else:
            self._cluster_input.setValue(self._last_clustered_value or 100)

    def reset_view(self):
        """Reset the 3D view to a default camera position."""
        if self._mode == "3D":
            if self.t1 is not None:
                self._3D_camera.show_object(self._3D_actors["t1"], (0, 0, -1))
            elif self.tractogram:
                if self._3D_actors["tractogram"] is not None:
                    self._3D_camera.show_object(
                        self._3D_actors["tractogram"], (0, 0, -1)
                    )
                else:
                    group = Group()
                    for bundle in self._streamline_bundles:
                        group.add(bundle)
                    for cluster in self._cluster_reps.values():
                        group.add(cluster)
                    self._3D_scene.add(group)
                    self._3D_camera.show_object(group, (0, 0, -1))
                    self._3D_scene.remove(group)
                    for cluster in self._cluster_reps.values():
                        self._3D_scene.add(cluster)
                    for bundle in self._streamline_bundles:
                        self._3D_scene.add(bundle)
            elif self.mesh:
                self._3D_camera.show_object(self._3D_actors["mesh"], (0, 0, -1))
            else:
                logging.warning("No object to center the view on.")
        self.show_manager.render()

    def toggle_suggestion(self):
        """Toggle suggestion mode."""
        if self._keystroke_card in self._3D_scene.ui_scene.children:
            self._3D_scene.ui_scene.remove(self._keystroke_card)
        else:
            self._3D_scene.ui_scene.add(self._keystroke_card)
