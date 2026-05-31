from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from dipy.io.stateful_tractogram import Space, StatefulTractogram
import numpy as np

from fury import distinguishable_colormap
from tractome.io import save_tractogram
from tractome.mem import input_manager, state_manager, visualization_manager
from tractome.ui._control_section import LeftSectionWidget
from tractome.ui._input_section import RightSectionWidget
from tractome.ui._paths import IMAGES_PATH
from tractome.ui._visualization_section import CenterSectionWidget
from tractome.ui.utils import open_file_dialog, save_file_dialog
from tractome.viz import create_streamlines, rasterize_box, rasterize_sphere


class StartScreen(QWidget):
    """Start screen of the app."""

    def __init__(self, on_uploading_done):
        """Initialize the start screen.

        Parameters
        ----------
        on_uploading_done : callable
            Callback invoked with the selected tractogram path.
        """
        super().__init__()

        self._on_uploading_done = on_uploading_done

        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)

        self._container_box = QFrame()
        self._container_box.setObjectName("startScreenContainer")
        self._container_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        inner_layout = QVBoxLayout(self._container_box)
        inner_layout.setAlignment(Qt.AlignCenter)
        inner_layout.setSpacing(20)

        self._logo_label = QLabel()
        logo_pixmap = QPixmap(str(IMAGES_PATH / "logo.png"))
        scaled_logo = logo_pixmap.scaled(
            127, 35, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        self._logo_label.setPixmap(scaled_logo)
        self._logo_label.setAlignment(Qt.AlignCenter)

        inner_layout.addStretch()
        inner_layout.addWidget(self._logo_label)

        self._title_label = QLabel("T R A C T O M E")
        self._title_label.setObjectName("titleLabel")
        self._title_label.setAlignment(Qt.AlignCenter)

        inner_layout.addWidget(self._title_label)

        self._upload_button = QPushButton("UPLOAD TRACTOGRAM")
        self._upload_button.setObjectName("startUploadButton")
        self._upload_button.setFixedSize(260, 50)
        self._upload_button.setCursor(Qt.PointingHandCursor)
        self._upload_button.clicked.connect(self._on_upload_clicked)

        inner_layout.addSpacing(40)
        inner_layout.addWidget(self._upload_button, alignment=Qt.AlignCenter)
        inner_layout.addStretch()

        layout.addWidget(self._container_box)

    def _on_upload_clicked(self):
        """Handle the upload button click event."""
        file_path = open_file_dialog(
            title="Select a tractogram file",
            file_filter=(
                "Tractogram Files (*.trx *.trk);; TRX Files (*.trx);; "
                "TRK Files (*.trk);; All Files (*.*)"
            ),
        )
        if file_path:
            print(f"Selected file: {file_path}")
            self._on_uploading_done(file_path)


class InteractionScreen(QWidget):
    """Interaction screen of the app."""

    change_tractogram_requested = Signal()

    def __init__(self):
        """Initialize the interaction screen."""
        super().__init__()

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        self._left_section = LeftSectionWidget(parent=self)
        self._center_section = CenterSectionWidget()
        self._right_section = RightSectionWidget()
        self._left_section.change_tractogram_requested.connect(
            self.change_tractogram_requested.emit
        )

        # Per-track color generator: each captured view gets the next
        # distinguishable color so neighbouring tracks don't collide.
        self._track_color_gen = distinguishable_colormap()
        self._right_section.image_input_widget.t1_changed.connect(self._on_t1_changed)
        self._right_section.image_input_widget.t1_visibility_changed.connect(
            self._on_t1_visibility_changed
        )
        self._right_section.image_input_widget.t1_slices_changed.connect(
            self._on_t1_slices_changed
        )
        self._right_section.mesh_input_widget.mesh_changed.connect(
            self._on_mesh_changed
        )
        self._right_section.mesh_input_widget.mesh_visibility_changed.connect(
            self._on_mesh_visibility_changed
        )
        self._right_section.mesh_input_widget.mesh_opacity_changed.connect(
            self._on_mesh_opacity_changed
        )
        self._right_section.mesh_input_widget.mesh_material_changed.connect(
            self._on_mesh_material_changed
        )
        self._right_section.mesh_input_widget.mesh_projection_changed.connect(
            self._on_mesh_projection_changed
        )
        self._right_section.mesh_input_widget.mesh_projection_threshold_changed.connect(
            self._on_mesh_projection_threshold_changed
        )
        self._right_section.parcel_input_widget.parcel_changed.connect(
            self._on_parcel_changed
        )
        self._right_section.parcel_input_widget.parcel_visibility_changed.connect(
            self._on_parcel_visibility_changed
        )
        self._right_section.parcel_input_widget.parcel_size_changed.connect(
            self._on_parcel_size_changed
        )
        self._left_section.roi_input_widget.rois_changed.connect(self._on_rois_changed)
        self._left_section.roi_input_widget.roi_visibility_changed.connect(
            self._on_roi_visibility_changed
        )
        self._left_section.roi_input_widget.roi_opacity_changed.connect(
            self._on_roi_opacity_changed
        )
        self._left_section.roi_input_widget.roi_create_requested.connect(
            self._on_roi_create_requested
        )
        self._left_section.roi_create_widget.shape_changed.connect(
            self._on_roi_create_shape_changed
        )
        self._left_section.roi_create_widget.finish_requested.connect(
            self._on_roi_create_finish_requested
        )
        self._left_section.roi_create_widget.edit_requested.connect(
            self._on_roi_create_edit_requested
        )
        self._left_section.roi_create_widget.roi_visibility_changed.connect(
            self._on_roi_create_visibility_changed
        )
        self._left_section.roi_create_widget.roi_remove_requested.connect(
            self._on_roi_create_remove_requested
        )
        self._center_section.roi_drawn.connect(self._on_roi_drawn)
        self._draft_roi_id = None
        self._roi_shape_by_id = {}
        self._left_section.view_mode_widget.view_mode_changed.connect(
            self._on_view_mode_changed
        )
        self._left_section.fibers_box.btn_capture.clicked.connect(
            self._on_capture_clicked
        )
        self._right_section.tracks_widget.track_visibility_changed.connect(
            self._on_track_visibility_changed
        )
        self._right_section.tracks_widget.track_save_requested.connect(
            self._on_track_save_requested
        )
        self._right_section.tracks_widget.track_remove_requested.connect(
            self._on_track_remove_requested
        )
        self._right_section.btn_toggle_info.clicked.connect(
            self._center_section.toggle_display_info
        )
        self._right_section.btn_toggle_shortcuts.clicked.connect(
            self._center_section.toggle_keystroke_card
        )

        main_layout.addWidget(self._left_section, 1)
        main_layout.addWidget(self._center_section, 3)
        main_layout.addWidget(self._right_section, 1)

        visualization_manager.set_wgpu_device(self._center_section.show_manager.device)

    def _on_t1_changed(self):
        """Refresh T1 visualization and reset slices for current T1."""
        if visualization_manager.t1_visualizations:
            self.remove_visualization(
                visualization_manager.t1_visualizations, visualization_type="t1"
            )

        t1_visualization = visualization_manager.visualize_t1()
        if t1_visualization is not None:
            self.add_visualization(t1_visualization, visualization_type="t1")
            self._right_section.image_input_widget.configure_t1_slice_controls()
            self._right_section.image_input_widget.emit_current_slices()
        self._right_section.image_input_widget.sync_t1_visibility_button()

        # The 2D scene caches its own T1/ROI/projection actors built
        # from the previously-current T1; without this rebuild a subject
        # change made while in 2D mode keeps the old image on screen
        # until the user toggles 3D→2D (which calls the same rebuild
        # via _on_view_mode_changed). The new T1's bounding box also
        # differs, so the camera needs to be re-framed against the new
        # actor, otherwise it would still be sized for the previous one.
        if state_manager.view_mode == "2D":
            self._build_2d_scene_contents()
            self._center_section.orient_2d_camera_to_active_slice()
            self._center_section.show_manager.render()

    def _on_t1_visibility_changed(self):
        """Re-render after toggling T1 visibility in the scene.

        In 2D mode the same signal also fires when the user picks a new
        active axis via the radio buttons, so the orthographic camera is
        re-aimed at the new slice plane before rendering. Without this
        the camera keeps the orientation it had when 2D mode was first
        entered and the new plane shows up edge-on.
        """
        if state_manager.view_mode == "2D":
            self._center_section.orient_2d_camera_to_active_slice()
        self._center_section.show_manager.render()

    def _on_t1_slices_changed(self, x, y, z):
        """Update shown T1 slices from the input controls.

        Parameters
        ----------
        x : int
            X slice position.
        y : int
            Y slice position.
        z : int
            Z slice position.
        """
        visualization_manager.show_t1_slices(x, y, z)
        self._center_section.show_manager.render()

    def _on_mesh_changed(self):
        """Reload mesh actor when the mesh/texture pair changes."""
        mesh_viz = visualization_manager.mesh_visualizations
        if mesh_viz:
            self.remove_visualization(mesh_viz, visualization_type="mesh")
        mesh_vis = visualization_manager.visualize_mesh()
        if mesh_vis is not None:
            self.add_visualization(mesh_vis, visualization_type="mesh")
        self._right_section.mesh_input_widget.sync_mesh_visibility_button()

        proj_viz = visualization_manager.mesh_projection_visualizations
        if proj_viz:
            self.remove_visualization(proj_viz, visualization_type="mesh_projection")
        visualization_manager.clear_mesh_projection()
        if state_manager.mesh_project:
            self._on_mesh_projection_changed(True)

    def _on_mesh_visibility_changed(self):
        """Re-render after toggling mesh visibility."""
        self._center_section.show_manager.render()

    def _on_mesh_opacity_changed(self, value):
        """Apply mesh opacity from the slider.

        Parameters
        ----------
        value : int
            Slider value in the range 0-100.
        """
        visualization_manager.set_mesh_opacity(value)
        self._center_section.show_manager.render()

    def _on_mesh_material_changed(self):
        """Rebuild mesh when the Photographic toggle changes."""
        if not input_manager.has_mesh:
            return
        mesh_viz = visualization_manager.mesh_visualizations
        if mesh_viz:
            self.remove_visualization(mesh_viz, visualization_type="mesh")
        mesh_vis = visualization_manager.visualize_mesh()
        if mesh_vis is not None:
            self.add_visualization(mesh_vis, visualization_type="mesh")
        self._right_section.mesh_input_widget.sync_mesh_visibility_button()

    def _refresh_mesh_projection_if_active(self):
        """Rebuild the projection actor from the current tractogram state.

        No-op when Project is off or prerequisites are missing. Called from
        any handler that mutates which streamlines are active or how they're
        coloured (ROI filter, re-cluster, cluster visibility, etc.).
        """
        if not state_manager.mesh_project:
            return
        old_viz, new_viz = visualization_manager.rebuild_mesh_projection(
            self._active_track_projection_source()
        )
        if old_viz:
            self.remove_visualization(old_viz, visualization_type="mesh_projection")
        if new_viz is None:
            return
        self.add_visualization(new_viz, visualization_type="mesh_projection")
        visualization_manager.update_mesh_projection(
            state_manager.mesh_projection_threshold
        )
        self._center_section.show_manager.render()

    def _on_mesh_projection_changed(self, enabled):
        """Toggle GPU projection of streamline points onto the mesh surface.

        While projection is active the tractogram actors are removed from
        the scene; they're re-added when projection is turned off.

        Parameters
        ----------
        enabled : bool
            Whether mesh projection should be active.
        """
        if not enabled:
            old_viz = visualization_manager.mesh_projection_visualizations
            if old_viz:
                self.remove_visualization(old_viz, visualization_type="mesh_projection")
            visualization_manager.clear_mesh_projection()
            tractogram_viz = visualization_manager.tractogram_visualizations
            if tractogram_viz:
                self.add_visualization(tractogram_viz, visualization_type="tractogram")
            self._apply_track_isolation()
            self._sync_keystroke_lock()
            self._center_section.show_manager.render()
            # Re-raise overlays above the freshly drawn canvas so the restored
            # keystroke card repaints now instead of on the next input event.
            self._center_section.refresh_overlays()
            return

        if not self._has_projectable_streamlines():
            self._show_projection_empty_warning()
            # Revert the checkbox without re-triggering this handler.
            checkbox = self._right_section.mesh_input_widget.project_checkbox
            checkbox.blockSignals(True)
            checkbox.setChecked(False)
            checkbox.blockSignals(False)
            state_manager.mesh_project = False
            self._right_section.mesh_input_widget._projection_controls.setVisible(False)
            self._sync_keystroke_lock()
            return

        # Always rebuild from current state on enable: cluster colors, ROI
        # filter, or visibility may have changed while Project was off.
        old_viz, new_viz = visualization_manager.rebuild_mesh_projection(
            self._active_track_projection_source()
        )
        if old_viz:
            self.remove_visualization(old_viz, visualization_type="mesh_projection")
        if new_viz is None:
            # Missing mesh, tractogram, or device — nothing to show.
            return
        # Lock keystrokes/card BEFORE detaching the tractogram: removing a
        # "tractogram" visualization hides the keystroke card as a side effect,
        # so the lock must snapshot the card's real pre-projection visibility
        # here to restore it correctly when projection is turned off.
        self._sync_keystroke_lock()
        tractogram_viz = visualization_manager.tractogram_visualizations
        if tractogram_viz:
            self.remove_visualization(tractogram_viz, visualization_type="tractogram")
        # add_visualization renders, which materializes the GPU buffer
        # so we can bind it as compute output on the next call.
        self.add_visualization(new_viz, visualization_type="mesh_projection")
        self._apply_track_isolation()
        visualization_manager.update_mesh_projection(
            state_manager.mesh_projection_threshold
        )
        self._center_section.show_manager.render()

    def _has_expanded_cluster(self):
        """Return True if any cluster is currently expanded."""
        if not state_manager.has_states():
            return False
        tractogram_states = state_manager.get_latest_state().tractogram_states
        if tractogram_states is None:
            return False
        return any(
            cluster_data.get("expanded") for cluster_data in tractogram_states.values()
        )

    def _active_track_projection_source(self):
        """Return streamline colors for checked captured tracks, or None."""
        streamline_colors = {}
        for track in self._right_section.tracks_widget.iter_tracks():
            if not track["visible"]:
                continue
            color = track["color"]
            for sid in track["streamline_ids"]:
                streamline_colors[int(sid)] = color
        return streamline_colors or None

    def _has_projectable_streamlines(self):
        """Return True if projection has a cluster or captured-track source."""
        return (
            self._active_track_projection_source() is not None
            or self._has_expanded_cluster()
        )

    def _show_projection_empty_warning(self):
        """Show the styled warning when no streamlines can be projected."""
        box = QMessageBox(self)
        box.setObjectName("captureWarningBox")
        box.setIcon(QMessageBox.Information)
        box.setWindowTitle("Nothing to project")
        box.setText(
            "Expand a cluster or select a captured track to visualize the projection."
        )
        box.setStandardButtons(QMessageBox.Ok)
        box.exec()

    def _on_mesh_projection_threshold_changed(self, threshold):
        """Re-dispatch the projection compute pass with a new threshold.

        Parameters
        ----------
        threshold : float
            Maximum projection distance.
        """
        if not state_manager.mesh_project:
            return
        if visualization_manager.mesh_projection_visualizations is None:
            return
        visualization_manager.update_mesh_projection(threshold)
        self._center_section.show_manager.render()

    def _on_parcel_changed(self):
        """Reload parcel actor when the file selection changes."""
        parcel_viz = visualization_manager.parcel_visualizations
        if parcel_viz:
            self.remove_visualization(parcel_viz, visualization_type="parcel")
        parcel_vis = visualization_manager.visualize_parcel()
        if parcel_vis is not None:
            self.add_visualization(parcel_vis, visualization_type="parcel")
        self._right_section.parcel_input_widget.sync_parcel_visibility_button()

    def _on_parcel_visibility_changed(self):
        """Re-render after toggling parcel visibility."""
        self._center_section.show_manager.render()

    def _on_parcel_size_changed(self, value):
        """Apply parcel point size from the slider.

        Parameters
        ----------
        value : int
            Slider value in the range 0-100. The UI labels this as opacity.
        """
        visualization_manager.set_parcel_size(value)
        self._center_section.show_manager.render()

    def _on_rois_changed(self):
        """Rebuild the ROI visualization and re-filter streamlines.

        Also re-clusters the tractogram on the filtered set when the
        filter changes so the scene reflects the new ROI selection.
        """
        roi_viz = list(visualization_manager.roi_visualizations)
        if roi_viz:
            self.remove_visualization(roi_viz, visualization_type="roi")
        roi_vis = visualization_manager.visualize_rois()
        if roi_vis:
            self.add_visualization(roi_vis, visualization_type="roi")
        self._left_section.roi_input_widget.refresh_rois()

        if visualization_manager.apply_roi_filter():
            tractogram_viz = list(visualization_manager.tractogram_visualizations or [])
            if tractogram_viz:
                self.remove_visualization(
                    tractogram_viz, visualization_type="tractogram"
                )
            nb_clusters = (
                state_manager.get_latest_state().nb_clusters
                if state_manager.has_states()
                else 100
            )
            tractogram_vis = visualization_manager.visualize_tractogram(
                nb_clusters=nb_clusters
            )
            if tractogram_vis is not None:
                self.add_visualization(tractogram_vis, visualization_type="tractogram")
            self._refresh_mesh_projection_if_active()

        if state_manager.view_mode == "2D":
            self._build_2d_scene_contents()
            self._center_section.orient_2d_camera_to_active_slice()
            self._left_section.update_controls_for_visualization()

    def _on_roi_visibility_changed(self):
        """Re-render after toggling per-ROI visibility."""
        self._center_section.show_manager.render()

    def _on_roi_opacity_changed(self, _value):
        """Re-render after the ROI opacity slider moves.

        Parameters
        ----------
        _value : int
            Slider value in the range 0-100.
        """
        self._center_section.show_manager.render()

    def _on_roi_create_requested(self):
        """Switch to 2D mode and start the interactive ROI draw."""
        if not input_manager.has_t1:
            self._show_reference_image_required_warning()
            return
        if state_manager.view_mode != "2D":
            self._left_section.view_mode_widget.set_mode("2D")
            self._on_view_mode_changed("2D")
        shape = self._left_section.roi_create_widget.current_shape()
        self._center_section.enter_roi_create_mode(shape)
        self._draft_roi_id = None
        self._left_section.update_controls_for_visualization()
        # Re-entering create mode shows every ROI drawn in earlier
        # sessions; the user can pick one to edit instead of starting
        # a new ROI from scratch.
        self._left_section.roi_create_widget.clear_existing_selection()
        self._refresh_roi_create_existing_list()
        QTimer.singleShot(0, self._refresh_roi_create_view)
        QTimer.singleShot(50, self._refresh_roi_create_view)

    def _refresh_roi_create_view(self):
        """Re-frame the 2D slice after ROI-create layout changes settle."""
        visualization_manager.toggle_t1_slice_visibility_2d(
            *state_manager.t1_slice_visibility_2d
        )
        self._center_section.orient_2d_camera_to_active_slice()
        self._center_section.show_manager.render()

    def _refresh_roi_create_existing_list(self):
        """Push the current set of ROIs into the create panel's list.

        The list mirrors ``input_manager.provided_roi_paths`` and is
        called whenever the set of ROIs changes (entering create
        mode, after each draw, after an edit-target change). Color
        comes from the visualization manager so the swatches in the
        list match the contours in the 3D scene.
        """
        items = []
        for index, name in enumerate(input_manager.provided_roi_paths):
            color = visualization_manager.get_roi_color(index)
            items.append({"name": str(name), "color": color})
        self._left_section.roi_create_widget.refresh_existing_rois(items)

    def _on_roi_create_shape_changed(self, shape):
        """Update the active shape used by the next drag.

        The draft pointer is intentionally **kept**: switching from
        sphere to rectangle (or back) on the same draft replaces that
        ROI's volume with the new primitive on the next drag, rather
        than spawning a third entry. To start a fresh ROI the user
        clears the existing-ROIs selection.
        """
        self._center_section.set_roi_create_shape(shape)

    def _on_roi_create_finish_requested(self):
        """Exit ROI editing and return to normal 2D mode."""
        self._commit_roi_create_session()

    def _on_roi_create_edit_requested(self, name):
        """Make the selected existing ROI the active edit target.

        ``name`` is the synthetic id stored in input_manager; an empty
        string means "no selection — next drag creates a new ROI".
        Setting the draft pointer to an existing ROI is all we need:
        the next drag flows through ``update_roi_volume`` and rewrites
        that ROI in place. We also push the selected ROI's metadata
        into the Properties pane so the user sees what they're editing.
        """
        if not name:
            self._draft_roi_id = None
            self._left_section.roi_create_widget.reset_properties()
            return

        roi_paths = list(input_manager.provided_roi_paths)
        if name not in roi_paths:
            self._draft_roi_id = None
            self._left_section.roi_create_widget.reset_properties()
            return

        index = roi_paths.index(name)
        self._draft_roi_id = name
        try:
            volume, _, _, _ = input_manager.get_roi_at(index)
        except ValueError:
            volume = None
        if volume is not None:
            try:
                import numpy as np  # local: shadow not desired at module scope

                ix, iy, iz = np.where(volume > 0)
                voxel_pos = (ix.mean(), iy.mean(), iz.mean()) if len(ix) else None
            except Exception:
                voxel_pos = None
        else:
            voxel_pos = None
        # The synthetic id is now shape-agnostic ("ROI N"); the
        # actual shape is read from the side-table populated each
        # time a draft is rasterized.
        shape = self._roi_shape_by_id.get(name)
        type_ = shape.capitalize() if shape else "–"
        color = visualization_manager.get_roi_color(index)
        self._left_section.roi_create_widget.set_properties(
            name=name,
            visibility=visualization_manager.is_roi_visible_at(index),
            type_=type_,
            position=voxel_pos,
            color=color,
        )

    def _on_roi_create_visibility_changed(self, name):
        """Toggle an existing ROI from the ROI edit list."""
        roi_paths = list(input_manager.provided_roi_paths)
        if name not in roi_paths:
            return
        index = roi_paths.index(name)
        visualization_manager.toggle_roi_visibility_at(index)
        self._left_section.roi_input_widget.refresh_rois()
        self._refresh_roi_create_existing_list()
        if self._draft_roi_id == name:
            self._left_section.roi_create_widget.set_properties(
                visibility=visualization_manager.is_roi_visible_at(index)
            )
        self._center_section.show_manager.render()

    def _on_roi_create_remove_requested(self, name):
        """Remove an existing ROI from the ROI edit list."""
        roi_paths = list(input_manager.provided_roi_paths)
        if name not in roi_paths:
            return
        input_manager.remove_roi(roi_paths.index(name))
        if self._draft_roi_id == name:
            self._draft_roi_id = None
            self._left_section.roi_create_widget.reset_properties()
        self._on_rois_changed()
        if state_manager.view_mode == "2D":
            self._build_2d_scene_contents()
            self._center_section.orient_2d_camera_to_active_slice()
        self._refresh_roi_create_existing_list()
        self._left_section.update_controls_for_visualization()
        self._center_section.show_manager.render()

    def _commit_roi_create_session(self):
        """Finish a create-mode session: exit mode + apply filter.

        Single-ROI-per-session: any ROIs drawn in this session stay
        in the input manager; the draft pointer is cleared, the
        center-section drag handlers are removed, and the streamline
        filter + recluster runs once so the 3D tractogram view
        reflects whatever was drawn.
        """
        was_create_mode = state_manager.roi_create_mode is not None
        had_draft = self._draft_roi_id is not None
        self._draft_roi_id = None
        self._center_section.exit_roi_create_mode()
        self._left_section.roi_create_widget.reset_properties()
        self._left_section.update_controls_for_visualization()
        self._left_section.roi_input_widget.refresh_rois()
        if was_create_mode and had_draft and input_manager.has_roi:
            self._on_rois_changed()

    def _on_roi_drawn(self, world_start, world_end, shape):
        """Rasterize the drawn shape into a binary volume.

        ``world_start`` and ``world_end`` are the two endpoints of the
        user's drag in world coordinates. For ``sphere`` they're treated
        as opposite ends of a diameter; for ``rectangle`` they're the
        opposite corners of the rectangle's diagonal.

        Behaviour: while the user keeps drawing without clicking
        ``New`` or ``Save`` the same draft ROI is overwritten in
        place. Clicking ``New`` clears the draft pointer so the next
        drag lands as a fresh entry. ``Save`` is what triggers the
        filter + recluster pass; this method never runs that pipeline.
        """
        if not input_manager.has_t1:
            return
        t1_volume, affine, _, _ = input_manager.get_current_t1()
        shape_volume = t1_volume.shape[:3]

        slice_axis = None
        for index, value in enumerate(state_manager.t1_slice_visibility_2d):
            if value:
                slice_axis = index
                break

        world_start = np.asarray(world_start, dtype=np.float64)
        world_end = np.asarray(world_end, dtype=np.float64)

        if shape == "rectangle" and slice_axis is not None:
            spacing = float(np.linalg.norm(affine[:3, slice_axis]))
            volume = rasterize_box(
                shape_volume,
                affine,
                world_start,
                world_end,
                slice_axis,
                spacing,
            )
        else:
            world_center = (world_start + world_end) / 2.0
            world_radius = float(np.linalg.norm(world_end - world_start)) / 2.0
            if world_radius <= 0:
                return
            volume = rasterize_sphere(
                shape_volume,
                affine,
                world_center,
                world_radius,
            )

        if not np.any(volume):
            return

        if self._draft_roi_id is None:
            self._draft_roi_id = input_manager.add_roi_volume(
                volume, affine, label=shape
            )
        else:
            input_manager.update_roi_volume(self._draft_roi_id, volume, affine)

        self._roi_shape_by_id[self._draft_roi_id] = shape

        roi_viz = list(visualization_manager.roi_visualizations)
        if roi_viz:
            self._center_section.remove_visualization(roi_viz, visualization_type="roi")
        roi_vis = visualization_manager.visualize_rois()
        if roi_vis:
            self._center_section.add_visualization(roi_vis, visualization_type="roi")

        self._center_section.remove_2d_visualization(
            visualization_manager.roi_2d_visualizations
        )
        roi_2d = visualization_manager.visualize_rois_2d()
        if roi_2d:
            self._center_section.add_2d_visualization(roi_2d)

        self._center_section.show_manager.render()
        QTimer.singleShot(0, self._center_section.show_manager.render)

        try:
            ix, iy, iz = np.where(volume > 0)
            voxel_pos = (ix.mean(), iy.mean(), iz.mean())
        except Exception:
            voxel_pos = None
        roi_index = next(
            (
                i
                for i, p in enumerate(input_manager.provided_roi_paths)
                if p == self._draft_roi_id
            ),
            -1,
        )
        color = visualization_manager.get_roi_color(roi_index)
        self._left_section.roi_create_widget.set_properties(
            name=self._draft_roi_id or "–",
            visibility=visualization_manager.is_roi_visible_at(roi_index),
            type_=shape.capitalize() if shape else "–",
            position=voxel_pos,
            color=color,
        )
        # Newly-added ROIs need to appear in the existing-ROIs list
        # immediately; updates to an existing draft re-emit the same
        # name so the list still refreshes (cheap — just rebuilds N
        # rows in a QListWidget).
        self._refresh_roi_create_existing_list()

    def _on_capture_clicked(self):
        """Snapshot the streamlines from currently-expanded clusters.

        Captures the streamlines from every expanded cluster — i.e. the
        individual streamlines currently rendered as lines in the scene.
        A new track row is appended to the Tracks panel on the right
        with a uniform per-view color. If no cluster is expanded,
        prompts the user to expand one first.
        """
        if not state_manager.has_states():
            return
        latest_state = state_manager.get_latest_state()
        if latest_state.tractogram_states is None:
            return
        streamline_ids = []
        for cluster_data in latest_state.tractogram_states.values():
            if not cluster_data.get("expanded"):
                continue
            streamline_ids.extend(int(s) for s in cluster_data["streamline_ids"])
        if not streamline_ids:
            self._show_capture_empty_warning()
            return
        color = tuple(float(c) for c in next(self._track_color_gen))
        self._right_section.tracks_widget.add_track(
            streamline_ids=streamline_ids, color=color
        )

    def _show_capture_empty_warning(self):
        """Show the styled warning when no expanded cluster exists."""
        box = QMessageBox(self)
        box.setObjectName("captureWarningBox")
        box.setIcon(QMessageBox.Information)
        box.setWindowTitle("Nothing to capture")
        box.setText("Expand a cluster before capturing a view.")
        box.setStandardButtons(QMessageBox.Ok)
        box.exec()

    def _show_reference_image_required_warning(self):
        """Warn that 2D / ROI editing needs a reference image loaded."""
        box = QMessageBox(self)
        box.setObjectName("referenceImageRequiredBox")
        box.setIcon(QMessageBox.Information)
        box.setWindowTitle("Reference image required")
        box.setText("To enable 2D mode or ROI editing a reference image is required.")
        box.setStandardButtons(QMessageBox.Ok)
        box.exec()

    def _sync_keystroke_lock(self):
        """Gate keystrokes, the keystroke card, and the shortcut toggle.

        They are disabled while a captured track is isolated or while mesh
        projection is active, so projection mirrors the capture-view lock.
        Driven off the combined state so turning one off does not unlock
        while the other is still on.
        """
        locked = self._right_section.tracks_widget.has_active_track() or bool(
            state_manager.mesh_project
        )
        self._center_section.set_track_isolation_active(locked)
        self._right_section.btn_toggle_shortcuts.setDisabled(locked)

    def _on_track_visibility_changed(self):
        """Apply or release the captured-track isolation in the scene."""
        self._apply_track_isolation()
        is_active = self._right_section.tracks_widget.has_active_track()
        self._left_section.set_track_isolation_active(is_active)
        self._sync_keystroke_lock()
        if state_manager.mesh_project and not self._has_projectable_streamlines():
            self._disable_mesh_projection()
            self._center_section.show_manager.render()
            return
        self._refresh_mesh_projection_if_active()
        self._center_section.show_manager.render()

    def _disable_mesh_projection(self):
        """Turn off projection and restore the interactive tractogram."""
        checkbox = self._right_section.mesh_input_widget.project_checkbox
        checkbox.blockSignals(True)
        checkbox.setChecked(False)
        checkbox.blockSignals(False)
        state_manager.mesh_project = False
        self._right_section.mesh_input_widget._projection_controls.setVisible(False)

        old_viz = visualization_manager.mesh_projection_visualizations
        if old_viz:
            self.remove_visualization(old_viz, visualization_type="mesh_projection")
        visualization_manager.clear_mesh_projection()

        tractogram_viz = visualization_manager.tractogram_visualizations
        if tractogram_viz:
            self.add_visualization(tractogram_viz, visualization_type="tractogram")

    def _apply_track_isolation(self):
        """Swap between cluster view and uniform-color tract view actors.

        While any track is active, all cluster actors are hidden and
        each visible track gets its own lines actor painted uniformly
        in the track's color. When no track is active the track actors
        are removed and per-cluster visibility is restored from each
        cluster's stored ``visible`` flag.
        """
        if not state_manager.has_states():
            return
        latest_state = state_manager.get_latest_state()
        if latest_state.tractogram_states is None:
            return

        has_active = self._right_section.tracks_widget.has_active_track()

        for cluster_data in latest_state.tractogram_states.values():
            actor = cluster_data.get("rep_actor") or cluster_data.get("lines_actor")
            if actor is None:
                continue
            actor.visible = (not has_active) and bool(cluster_data.get("visible", True))

        scene = self._center_section._3D_scene
        for track in self._right_section.tracks_widget.iter_tracks():
            actor = track.get("actor")
            if track["visible"]:
                if actor is None:
                    actor = self._build_track_actor(track)
                    if actor is None:
                        continue
                    track["actor"] = actor
                    actor.visible = not state_manager.mesh_project
                    scene.add(actor)
                else:
                    actor.visible = not state_manager.mesh_project
            elif actor is not None:
                actor.visible = False

    def _build_track_actor(self, track):
        """Build a uniform-color streamlines actor for a captured track.

        Parameters
        ----------
        track : dict
            Captured-track metadata containing streamline ids and color.

        Returns
        -------
        Line or None
            Streamlines actor, or None if no tractogram/streamlines are available.
        """
        if not input_manager.has_tractogram:
            return None
        sft, _, _, _ = input_manager.get_current_tractogram()
        streamlines = [sft.streamlines[i] for i in track["streamline_ids"]]
        if not streamlines:
            return None
        color = np.asarray(track["color"], dtype=np.float32)
        return create_streamlines(streamlines, color)

    def _on_track_save_requested(self, index):
        """Save the streamlines belonging to a captured track as TRX.

        Parameters
        ----------
        index : int
            Index of the track to save.
        """
        track = self._right_section.tracks_widget.get_track(index)
        if track is None or not track["streamline_ids"]:
            return
        if not input_manager.has_tractogram:
            return
        file_path = save_file_dialog(
            parent=self,
            title="Save track as TRX",
            file_filter="TRX Files (*.trx);; All Files (*.*)",
            default_name=f"{track['name']}.trx",
        )
        if not file_path:
            return
        if not file_path.lower().endswith(".trx"):
            file_path = f"{file_path}.trx"
        sft, _, _, _ = input_manager.get_current_tractogram()
        selected = [sft.streamlines[i] for i in track["streamline_ids"]]
        new_sft = StatefulTractogram(selected, sft, Space.RASMM)
        save_tractogram(new_sft, file_path)

    def _on_track_remove_requested(self, index):
        """Remove a captured track and refresh the scene if it was active.

        Parameters
        ----------
        index : int
            Index of the track to remove.
        """
        track = self._right_section.tracks_widget.get_track(index)
        if track is not None:
            actor = track.get("actor")
            if actor is not None:
                self._center_section._3D_scene.remove(actor)
                track["actor"] = None
        self._right_section.tracks_widget.remove_track(index)
        self._on_track_visibility_changed()

    def _on_view_mode_changed(self, mode):
        """Switch the active scene and matching control panels for ``mode``.

        Building the 2D scene is deferred to the first 2D toggle: the
        T1, ROI, and projection actors required for orthographic viewing
        are only constructed when the user opts in.

        Parameters
        ----------
        mode : str
            Either ``"3D"`` or ``"2D"``.
        """
        if state_manager.view_mode == mode:
            return

        if mode == "2D" and not input_manager.has_t1:
            self._show_reference_image_required_warning()
            self._left_section.view_mode_widget.set_mode(state_manager.view_mode)
            return

        # Leaving create mode for 3D commits the session: any drawn
        # ROI is finalized, the panel resets, and the streamline
        # filter + recluster runs once so the 3D tractogram view
        # reflects the new ROI. This is the only place the filter
        # runs during a create session — per-draw refreshes never
        # touch the cluster pipeline.
        if state_manager.roi_create_mode is not None and mode == "3D":
            self._commit_roi_create_session()

        state_manager.view_mode = mode
        if mode == "2D":
            self._build_2d_scene_contents()
            self._right_section.mesh_input_widget.setVisible(False)
            self._right_section.parcel_input_widget.setVisible(False)
            self._right_section.image_input_widget.set_slice_control_mode("radio")
        else:
            self._right_section.mesh_input_widget.setVisible(True)
            self._right_section.parcel_input_widget.setVisible(True)
            self._right_section.image_input_widget.set_slice_control_mode("checkbox")

        self._center_section.set_view_mode(mode)
        self._left_section.update_controls_for_visualization()

    def _build_2d_scene_contents(self):
        """Rebuild the actors that belong to the 2D scene.

        Existing 2D T1 / ROI / streamline actors are removed first so the
        2D scene reflects the current data and selection. The streamline
        projections are recomputed from the latest cluster selection.
        """
        center = self._center_section
        center.remove_2d_visualization(visualization_manager.t1_2d_visualizations)
        center.remove_2d_visualization(visualization_manager.roi_2d_visualizations)
        center.remove_2d_visualization(
            visualization_manager.streamlines_2d_visualizations
        )

        t1_2d = visualization_manager.visualize_t1_2d()
        if t1_2d:
            center.add_2d_visualization(t1_2d)

        roi_2d = visualization_manager.visualize_rois_2d()
        if roi_2d:
            center.add_2d_visualization(roi_2d)

        projections = visualization_manager.visualize_streamlines_projection_2d()
        if projections:
            center.add_2d_visualization(projections)

    def add_visualization(self, visualizations, visualization_type="unknown"):
        """Add a visualization to the center section.

        Parameters
        ----------
        visualizations : list
            Visualizations to add.
        visualization_type : str, optional
            Type/category of the visualization payload.
        """
        self._center_section.add_visualization(
            visualizations, visualization_type=visualization_type
        )
        self._left_section.update_controls_for_visualization()
        self._center_section.show_manager.render()

    def remove_visualization(self, visualizations, *, visualization_type="unknown"):
        """Remove a visualization from the center section.

        Parameters
        ----------
        visualizations : list
            Visualizations to remove.
        """
        self._center_section.remove_visualization(
            visualizations, visualization_type=visualization_type
        )
        self._left_section.update_controls_for_visualization()
        self._center_section.show_manager.render()
