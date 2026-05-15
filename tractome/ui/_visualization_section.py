from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)
import numpy as np
import pylinalg as la

from fury import actor, window
from fury.lib import (
    DirectionalLight,
    Event,
    OrthographicCamera,
    PanZoomController,
    PerspectiveCamera,
    TrackballController,
)
from tractome.mem import state_manager, visualization_manager


class CenterSectionWidget(QFrame):
    """Center section container for visualization/content."""

    roi_drawn = Signal(object, object, str)

    def __init__(self):
        super().__init__()
        self.setObjectName("interactionCenterSection")
        self.setFrameShape(QFrame.StyledPanel)

        layout = QGridLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self._3D_scene = window.Scene(background=(0.2, 0.2, 0.2))
        self._3D_camera = PerspectiveCamera()
        self._3D_camera.add(DirectionalLight())
        self._3D_scene.add(self._3D_camera)
        self._3D_controller = TrackballController(self._3D_camera)

        self._2D_scene = window.Scene(background=(0.2, 0.2, 0.2))
        self._2D_scene.add(DirectionalLight())
        self._2D_camera = OrthographicCamera()
        self._2D_controller = PanZoomController(self._2D_camera)

        self.show_manager = window.ShowManager(
            scene=self._3D_scene,
            camera=self._3D_camera,
            controller=self._3D_controller,
            qt_app=QApplication.instance(),
            window_type="qt",
        )

        self._3D_controller.register_events(self.show_manager.renderer)
        self._2D_controller.register_events(self.show_manager.renderer)
        self._2D_controller.enabled = False

        # TODO: Remove long press event handler for Qt
        # This is a temporary workaround for the long press issue in Qt
        self.show_manager.renderer.remove_event_handler(
            self.show_manager._set_key_long_press_event, "key_up", "key_down"
        )

        self._roi_create_initial_pos = None
        self._roi_create_preview = None
        self._roi_drag_handlers_registered = False
        self._roi_create_dragging = False
        self._saved_2d_controls = None

        self._keystrokes_enabled = True
        self._keystroke_card_pre_isolation_visible = None

        def _register_clicks(event):
            """Handle selection clicks.

            Parameters
            ----------
            event : Event
                The click event.
            """
            if state_manager.roi_create_mode is not None:
                return
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

        self.show_manager.renderer.add_event_handler(
            self.handle_key_strokes, "key_down"
        )

        viz_window = self.show_manager.window
        if not isinstance(viz_window, QWidget):
            viz_window = QWidget.createWindowContainer(viz_window, self)

        viz_window.setObjectName("interactionVizWindow")
        layout.addWidget(viz_window, 0, 0)

        self._build_display_info_overlay(layout)
        self._build_keystroke_card(layout)

        self._axes_gizmo_margin = 60
        self.show_manager.show_axes_gizmo(size=30, thickness=2)
        # The wgpu canvas reports its own resize before Qt's
        # CenterSectionWidget.resizeEvent fires, so listen on both.
        self.show_manager.resize_callback(self._on_canvas_resize)
        QTimer.singleShot(0, self._reposition_axes_gizmo)

    def _build_display_info_overlay(self, parent_layout):
        """Create display info overlay inside the visualization area."""
        self._display_info_widget = QFrame(self)
        self._display_info_widget.setObjectName("displayInfoWidget")
        self._display_info_widget.setAttribute(Qt.WA_TransparentForMouseEvents, True)

        info_layout = QVBoxLayout(self._display_info_widget)
        info_layout.setContentsMargins(10, 10, 10, 10)
        info_layout.setSpacing(4)

        title = QLabel("Display info", self._display_info_widget)
        title.setObjectName("displayInfoTitle")
        info_layout.addWidget(title)

        self._display_cluster_count = QLabel(
            "# of clusters: 0", self._display_info_widget
        )
        self._display_roi_count = QLabel("# of ROI: 0", self._display_info_widget)
        self._display_fibers_count = QLabel("# of fibers: 0", self._display_info_widget)

        info_layout.addWidget(self._display_cluster_count)
        info_layout.addWidget(self._display_roi_count)
        info_layout.addWidget(self._display_fibers_count)

        parent_layout.addWidget(
            self._display_info_widget,
            0,
            0,
            alignment=Qt.AlignTop | Qt.AlignLeft,
        )
        self._display_info_widget.raise_()

    def _build_keystroke_card(self, parent_layout):
        """Build the keystroke card."""
        self._keystroke_card = QFrame(self)
        self._keystroke_card.setObjectName("keystrokeCard")
        self._keystroke_card.setAttribute(Qt.WA_TransparentForMouseEvents, True)

        info_layout = QVBoxLayout(self._keystroke_card)
        info_layout.setContentsMargins(10, 10, 10, 10)
        info_layout.setSpacing(4)

        title = QLabel("Key Strokes", self._keystroke_card)
        title.setObjectName("keystrokeTitle")
        info_layout.addWidget(title)

        shortcut_lines = [
            "a: Select All",
            "n: Select None",
            "i: Swap Selection",
            "d: Delete Selection",
            "e: Expand Selection",
            "c: Collapse Selection",
            "s: Show Selection",
            "h: Hide Selection",
            "x: Toggle this message",
        ]
        self._keystroke_content = QLabel(
            "\n".join(shortcut_lines), self._keystroke_card
        )
        self._keystroke_content.setObjectName("keystrokeContent")
        self._keystroke_content.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        info_layout.addWidget(self._keystroke_content)

        parent_layout.addWidget(
            self._keystroke_card,
            0,
            0,
            alignment=Qt.AlignBottom | Qt.AlignLeft,
        )
        self._keystroke_card.raise_()

    def add_visualization(self, visualizations, visualization_type="unknown"):
        """Add visualizations to the center section.

        Parameters
        ----------
        visualizations : list
            The visualizations to add.
        visualization_type : str, optional
            Type/category of the visualization payload.
        """
        self._3D_scene.add(*visualizations)
        if visualization_type == "tractogram":
            self._update_display_info()
            self._keystroke_card.setVisible(True)
        self._refresh_overlays()

    def resizeEvent(self, event):
        """Re-pin the axes gizmo to the bottom-right after a canvas resize."""
        super().resizeEvent(event)
        self._reposition_axes_gizmo()

    def showEvent(self, event):
        """Reposition once the widget is on-screen.

        QStackedWidget hides this view at app start, so the only resize
        Qt fires before show is for the (default) 800x800 canvas. Without
        this call the gizmo would otherwise sit at the unscaled default
        position until the first user resize.
        """
        super().showEvent(event)
        QTimer.singleShot(0, self._reposition_axes_gizmo)

    def _on_canvas_resize(self, size):
        """ShowManager resize callback that mirrors ``_reposition_axes_gizmo``.

        ``size`` is the new ``(width, height)`` of the wgpu canvas, in
        logical pixels — when there's a single screen filling the
        window it matches ``screens[0].size`` so we use it directly
        without having to wait for ``update_viewports`` to run.
        """
        anchor = getattr(self.show_manager, "_axes_helper_anchor", None)
        if anchor is None:
            return
        try:
            width, _height = size
        except (TypeError, ValueError):
            return
        margin = self._axes_gizmo_margin
        anchor.local.position = [max(int(width) - margin, margin), margin, 0.5]
        self.show_manager.render()

    def _reposition_axes_gizmo(self):
        """Move the axes-gizmo anchor to the bottom-right of the viewport.

        Fury anchors the helper at ``(60, 60)`` from the bottom-left by
        default; we keep the same vertical inset and mirror the horizontal
        one so the gizmo sticks to the bottom-right corner regardless of
        canvas size.
        """
        anchor = getattr(self.show_manager, "_axes_helper_anchor", None)
        if anchor is None:
            return
        screens = getattr(self.show_manager, "screens", None)
        if not screens:
            return
        try:
            width, _height = screens[0].size
        except Exception:
            return
        margin = self._axes_gizmo_margin
        anchor.local.position = [max(int(width) - margin, margin), margin, 0.5]
        self.show_manager.render()

    def toggle_display_info(self):
        """Flip visibility of the Display Info overlay card."""
        self._display_info_widget.setVisible(not self._display_info_widget.isVisible())
        self._refresh_overlays()

    def toggle_keystroke_card(self):
        """Flip visibility of the Key Strokes overlay card."""
        self._keystroke_card.setVisible(not self._keystroke_card.isVisible())
        self._refresh_overlays()

    def set_track_isolation_active(self, active):
        """Disable keystrokes and hide the keystroke card while a tract view is active.

        On entering isolation, the current keystroke-card visibility is
        remembered and restored when isolation ends. Keystroke handling
        is also gated so cluster-mutation keys (a/n/i/d/e/c/s/h/x) do
        nothing while a captured view is shown.
        """
        if active:
            if self._keystrokes_enabled:
                self._keystroke_card_pre_isolation_visible = (
                    self._keystroke_card.isVisible()
                )
                self._keystroke_card.setVisible(False)
                self._refresh_overlays()
            self._keystrokes_enabled = False
        else:
            if not self._keystrokes_enabled:
                if self._keystroke_card_pre_isolation_visible is not None:
                    self._keystroke_card.setVisible(
                        self._keystroke_card_pre_isolation_visible
                    )
                    self._refresh_overlays()
                self._keystroke_card_pre_isolation_visible = None
            self._keystrokes_enabled = True

    def remove_visualization(self, visualizations, *, visualization_type="unknown"):
        """Remove visualizations from the center section.

        Parameters
        ----------
        visualizations : list
            The visualizations to remove.
        """
        self._3D_scene.remove(*visualizations)
        if visualization_type == "tractogram":
            self._update_display_info()
            self._keystroke_card.setVisible(False)
        self._refresh_overlays()

    def add_2d_visualization(self, visualizations):
        """Add visualizations to the 2D scene.

        Parameters
        ----------
        visualizations : list
            The 2D actors to add.
        """
        if not visualizations:
            return
        self._2D_scene.add(*visualizations)

    def remove_2d_visualization(self, visualizations):
        """Remove visualizations from the 2D scene.

        Parameters
        ----------
        visualizations : list
            The 2D actors to remove.
        """
        if not visualizations:
            return
        self._2D_scene.remove(*visualizations)

    def set_view_mode(self, mode):
        """Swap the active scene/camera/controller for 3D or 2D mode.

        The renderer pipeline only commits the new scene on the frame
        after the swap, so a single ``render()`` call after switching
        leaves a stale image on the canvas. A deferred second render is
        scheduled via ``QTimer.singleShot`` to guarantee a fresh paint
        once Qt has processed the scene/camera reassignment.

        Parameters
        ----------
        mode : str
            Either ``"3D"`` or ``"2D"``.
        """
        screen = self.show_manager.screens[0]
        if mode == "2D":
            screen.scene = self._2D_scene
            screen.camera = self._2D_camera
            screen.controller = self._2D_controller
            self._3D_controller.enabled = False
            self._2D_controller.enabled = True
            self.orient_2d_camera_to_active_slice()
        else:
            screen.scene = self._3D_scene
            screen.camera = self._3D_camera
            screen.controller = self._3D_controller
            self._3D_controller.enabled = True
            self._2D_controller.enabled = False
        self.show_manager.render()
        QTimer.singleShot(0, self.show_manager.render)
        self._refresh_overlays()

    def _2D_scene_t1_actor(self):
        """Return the T1 slicer currently in the 2D scene, or None."""
        t1_viz = visualization_manager.t1_2d_visualizations
        if not t1_viz:
            return None
        return t1_viz[0]

    def orient_2d_camera_to_active_slice(self):
        """Aim the 2D orthographic camera down the currently active slice axis.

        Reads ``state_manager.t1_slice_visibility_2d`` (a 3-tuple of
        booleans, exactly one of which is True in 2D mode) and frames the
        T1 slicer so the visible plane faces the camera. Without this the
        camera keeps the orientation it had when 2D mode was first
        entered, so toggling the X/Y/Z radio leaves the camera looking
        edge-on at the new plane.

        ``up`` is passed explicitly because ``show_object`` otherwise
        keeps the camera's current up vector — and going e.g. X → Y, the
        previous up (+Y) ends up parallel to the new view direction
        (along Y). pygfx then has no valid right-axis to build the view
        basis and the slice renders off-frame / edge-on. The convention
        used here matches a standard slice viewer: axial (Z) views use
        +Y as up, sagittal (X) and coronal (Y) views use +Z.
        """
        t1_actor = self._2D_scene_t1_actor()
        if t1_actor is None:
            return
        visibility = np.asarray(state_manager.t1_slice_visibility_2d, dtype=int)
        if not np.any(visibility):
            return
        active_axis = int(np.argmax(visibility))
        view_direction = tuple((-1 * visibility).tolist())
        up = (0.0, 1.0, 0.0) if active_axis == 2 else (0.0, 0.0, 1.0)
        self._2D_camera.show_object(t1_actor, view_direction, up=up)

    def _refresh_overlays(self):
        """Force the Qt overlays to fully repaint after a scene update.

        The keystroke card sits over the wgpu canvas with a translucent
        background and its text never changes, so repeated scene frames
        otherwise leave antialiased ghosts on top of the static labels.
        Re-raising and updating the widgets clears the accumulated pixels.
        """
        for overlay in (self._display_info_widget, self._keystroke_card):
            overlay.raise_()
            overlay.update()

    def _update_display_info(self):
        """Update display info overlay for latest added visualization."""
        latest_state = state_manager.get_latest_state()

        cluster_count = latest_state.nb_clusters
        fibers_count = len(latest_state.streamline_ids)
        if latest_state.filtered_streamline_ids is not None:
            visible_ids = set(
                np.asarray(latest_state.streamline_ids, dtype=np.int64).tolist()
            )
            filtered_ids = set(
                np.asarray(
                    latest_state.filtered_streamline_ids, dtype=np.int64
                ).tolist()
            )
            roi_count = len(visible_ids & filtered_ids)
        else:
            roi_count = "N/A"

        self._display_cluster_count.setText(f"# of clusters: {cluster_count}")
        self._display_roi_count.setText(f"# of ROI: {roi_count}")
        self._display_fibers_count.setText(f"# of fibers: {fibers_count}")

    def enter_roi_create_mode(self, shape):
        """Enter the interactive ROI drawing mode for the given shape.

        Unbinds left-drag pan on the 2D controller (so the drag draws an
        ROI) while leaving wheel/right-drag zoom active, and registers
        drag handlers for drawing on the active slice plane.
        """
        state_manager.roi_create_mode = shape
        self._roi_create_initial_pos = None
        self._roi_create_dragging = False
        self._cleanup_roi_preview()
        if self._saved_2d_controls is None:
            self._saved_2d_controls = dict(self._2D_controller.controls)
            self._2D_controller.controls.pop("mouse1", None)
            self._2D_controller.controls.pop("mouse2", None)
        if not self._roi_drag_handlers_registered:
            self.show_manager.renderer.add_event_handler(
                self._on_roi_create_drag, "pointer_drag"
            )
            self.show_manager.renderer.add_event_handler(
                self._on_roi_create_release, "pointer_up"
            )
            self._roi_drag_handlers_registered = True

    def exit_roi_create_mode(self):
        """Leave the ROI create mode and restore normal interactions."""
        state_manager.roi_create_mode = None
        self._roi_create_initial_pos = None
        self._cleanup_roi_preview()
        self._roi_create_dragging = False
        if self._roi_drag_handlers_registered:
            self.show_manager.renderer.remove_event_handler(
                self._on_roi_create_drag, "pointer_drag"
            )
            self.show_manager.renderer.remove_event_handler(
                self._on_roi_create_release, "pointer_up"
            )
            self._roi_drag_handlers_registered = False
        if self._saved_2d_controls is not None:
            self._2D_controller.controls.clear()
            self._2D_controller.controls.update(self._saved_2d_controls)
            self._saved_2d_controls = None
        self.show_manager.render()

    def set_roi_create_shape(self, shape):
        """Update the active shape mid-mode without re-registering handlers."""
        if state_manager.roi_create_mode is None:
            return
        state_manager.roi_create_mode = shape

    def _cleanup_roi_preview(self):
        """Remove the preview disk actor from the 2D scene if present."""
        if self._roi_create_preview is not None:
            try:
                self._2D_scene.remove(self._roi_create_preview)
            except Exception:
                pass
            self._roi_create_preview = None

    def _active_2d_slice_axis(self):
        """Return the voxel-axis index visible in the 2D scene, or None."""
        visibility = state_manager.t1_slice_visibility_2d
        for index, value in enumerate(visibility):
            if value:
                return index
        return None

    def _slice_plane_world(self, axis):
        """Return ``(plane_point, plane_normal)`` for the active slice plane.

        ``state_manager.t1_state`` already stores the slice positions in
        world coordinates (the sliders are populated from the T1 actor's
        world-space bounding box), so we use it directly as a point on
        the plane. The slicer renders axis-aligned planes in world
        space, so the normal is simply the world unit axis matching
        the active slice index.
        """
        plane_point = np.asarray(state_manager.t1_state, dtype=np.float64)
        normal = np.zeros(3, dtype=np.float64)
        normal[axis] = 1.0
        return plane_point, normal

    def _screen_to_slice_world(self, x, y):
        """Convert canvas pixel ``(x, y)`` to a world point on the active slice.

        Uses pygfx's own ``pylinalg.vec_transform`` so that NDC z and
        homogeneous division follow pygfx's WebGPU clip-space
        convention (z_ndc in [0, 1], not OpenGL's [-1, 1]). The unprojected
        near and far points define the camera ray; we then intersect that
        ray with the active slice plane in world space.
        """
        axis = self._active_2d_slice_axis()
        if axis is None:
            return None, None
        plane_point, plane_normal = self._slice_plane_world(axis)

        screen = self.show_manager.screens[0]
        try:
            vp_x, vp_y, vp_w, vp_h = screen.viewport.rect
        except Exception:
            vp_w, vp_h = self.show_manager.size
            vp_x, vp_y = 0, 0
        if vp_w <= 0 or vp_h <= 0:
            return None, None

        x_local = float(x) - float(vp_x)
        y_local = float(y) - float(vp_y)
        x_ndc = (2.0 * x_local / float(vp_w)) - 1.0
        y_ndc = 1.0 - (2.0 * y_local / float(vp_h))

        camera = self._2D_camera
        proj_inv = np.asarray(camera.projection_matrix_inverse, dtype=np.float64)
        cam_world = np.asarray(camera.world.matrix, dtype=np.float64)

        # WebGPU clip-space: near plane at z_ndc=0, far plane at z_ndc=1.
        ndc_near = np.array([x_ndc, y_ndc, 0.0], dtype=np.float64)
        ndc_far = np.array([x_ndc, y_ndc, 1.0], dtype=np.float64)
        view_near = la.vec_transform(ndc_near, proj_inv)
        view_far = la.vec_transform(ndc_far, proj_inv)
        world_near = la.vec_transform(view_near, cam_world)
        world_far = la.vec_transform(view_far, cam_world)

        ray_dir = world_far - world_near
        norm = float(np.linalg.norm(ray_dir))
        if norm < 1e-9:
            return None, None
        ray_dir = ray_dir / norm
        denom = float(np.dot(ray_dir, plane_normal))
        if abs(denom) < 1e-6:
            return None, None
        t = float(np.dot(plane_point - world_near, plane_normal)) / denom
        world_point = world_near + t * ray_dir
        return world_point, axis

    def _ensure_preview_disk(self, axis):
        """Create the preview disk lazily once a gesture has two points.

        The disk is unit-radius, translucent, lifted off the slice
        along the slice plane normal so it isn't z-fighting the
        slicer, and given a low ``render_order`` (this codebase's
        weighted_blend slicer composites cleanest when the preview
        sits below the slicer in the sort order).
        """
        if self._roi_create_preview is not None:
            return self._roi_create_preview
        _, plane_normal = self._slice_plane_world(axis)
        self._roi_create_plane_normal = np.asarray(plane_normal, dtype=np.float32)
        # Build the disk with default ``directions`` so its vertices
        # land in the XY plane (normal = +Z). FURY's repeat_primitive
        # treats X (not Z) as the source normal when applying the
        # ``directions`` rotation, so passing ``(0,0,1)`` for a Z-slice
        # tilts the disk onto the YZ plane and renders it edge-on to
        # the camera (i.e. invisible). We instead reorient via
        # ``local.rotation`` below.
        preview = actor.disk(
            np.zeros((1, 3), dtype=np.float32),
            colors=(0.3, 0.7, 1.0),
            radii=1.0,
            opacity=0.5,
            material="basic",
        )
        try:
            q = la.quat_from_vecs(
                np.array([0.0, 0.0, 1.0], dtype=np.float64),
                np.asarray(plane_normal, dtype=np.float64),
            )
            preview.local.rotation = q
        except Exception:
            pass
        preview.material.alpha_mode = "weighted_blend"
        preview.material.depth_write = False
        self._roi_create_preview = preview
        self._2D_scene.add(preview)
        return preview

    def _update_preview_disk(self, center, radius):
        """Position and scale the preview disk for the current drag state.

        Offset by one world unit along the slice plane normal so the
        disk's depth differs from the slicer's depth — same-depth
        produced an invisible result on the user's setup.
        """
        preview = self._roi_create_preview
        if preview is None or radius <= 0:
            return
        normal = getattr(
            self, "_roi_create_plane_normal", np.array([0, 0, 1], np.float32)
        )
        center = np.asarray(center, dtype=np.float32) + normal * 1.0
        preview.local.position = tuple(center)
        preview.local.scale = (float(radius), float(radius), 1.0)

    def _ensure_preview_rect(self, axis):
        """Create the rubber-band preview square for the rectangle tool.

        Same lifecycle and styling as the disk preview: a unit square
        in the local XY plane, rotated to align with the active slice
        plane, and scaled non-uniformly each drag tick to match the
        in-plane bounds of the user's drag.
        """
        if self._roi_create_preview is not None:
            return self._roi_create_preview
        _, plane_normal = self._slice_plane_world(axis)
        self._roi_create_plane_normal = np.asarray(plane_normal, dtype=np.float32)
        preview = actor.square(
            np.zeros((1, 3), dtype=np.float32),
            colors=(0.3, 0.7, 1.0),
            scales=(1.0, 1.0, 1.0),
            opacity=0.5,
            material="basic",
        )
        try:
            q = la.quat_from_vecs(
                np.array([0.0, 0.0, 1.0], dtype=np.float64),
                np.asarray(plane_normal, dtype=np.float64),
            )
            preview.local.rotation = q
        except Exception:
            pass
        preview.material.alpha_mode = "weighted_blend"
        preview.material.depth_write = False
        self._roi_create_preview = preview
        self._2D_scene.add(preview)
        return preview

    def _update_preview_rect(self, start, end, axis):
        """Position and scale the preview rectangle for the current drag.

        ``actor.square`` produces a unit quad in the local XY plane.
        The rotation set at creation maps that local plane onto the
        slice plane; for the three axis-aligned slice normals produced
        by ``_slice_plane_world`` the mapping is:

        * axis 0 (X-slice): local X ↔ world Z, local Y ↔ world Y
        * axis 1 (Y-slice): local X ↔ world X, local Y ↔ world Z
        * axis 2 (Z-slice): local X ↔ world X, local Y ↔ world Y

        Sign of the mapping doesn't matter for the scale; we use the
        absolute extents along the relevant world axes.
        """
        preview = self._roi_create_preview
        if preview is None:
            return
        a = np.asarray(start, dtype=np.float64)
        b = np.asarray(end, dtype=np.float64)
        delta = np.abs(b - a)
        if axis == 0:
            scale_x, scale_y = float(delta[2]), float(delta[1])
        elif axis == 1:
            scale_x, scale_y = float(delta[0]), float(delta[2])
        else:
            scale_x, scale_y = float(delta[0]), float(delta[1])
        if scale_x <= 0 or scale_y <= 0:
            return
        normal = getattr(
            self, "_roi_create_plane_normal", np.array([0, 0, 1], np.float32)
        )
        center = ((a + b) / 2.0).astype(np.float32) + normal * 1.0
        preview.local.position = tuple(center)
        preview.local.scale = (scale_x, scale_y, 1.0)

    def _on_roi_create_drag(self, event):
        """Track the drag and update the live ROI preview.

        Sphere tool: the click point is one end of the diameter and
        the current pointer is the other; the disk preview is the
        midpoint-centred circle of half that distance.

        Rectangle tool: the click point and current pointer are
        opposite corners of the rectangle's diagonal; the preview is
        the axis-aligned rubber-band quad bounding them on the slice
        plane.

        The first event of a gesture only records the start point;
        the preview is created on the second event so it's already at
        a visible size from frame one.

        FURY synthesizes ``pointer_drag`` from ``pointer_down`` +
        ``pointer_move`` (see ShowManager._register_drag in window.py).
        """
        if state_manager.roi_create_mode is None:
            return
        world_pos, axis = self._screen_to_slice_world(event.x, event.y)
        if world_pos is None:
            return
        if self._roi_create_initial_pos is None:
            self._roi_create_initial_pos = (world_pos, axis)
            self._roi_create_dragging = True
            return

        start, fixed_axis = self._roi_create_initial_pos
        end = world_pos
        if state_manager.roi_create_mode == "rectangle":
            self._ensure_preview_rect(fixed_axis)
            self._update_preview_rect(start, end, fixed_axis)
        else:
            center = (np.asarray(start) + np.asarray(end)) / 2.0
            radius = float(np.linalg.norm(end - start)) / 2.0
            if radius <= 0:
                return
            self._ensure_preview_disk(fixed_axis)
            self._update_preview_disk(center, radius)
        # During the drag we are inside a Qt mouse-event handler, so a
        # plain ``request_draw`` only schedules the paint for the next
        # vsync — but Qt is busy delivering more mouse events, so the
        # paint event never fires until we let go. ``force_draw`` calls
        # the canvas's ``repaint`` synchronously, and we follow it with
        # a ``processEvents`` so Qt actually delivers the paint event
        # before the next pointer event arrives. ``render`` first so
        # the canvas has a draw function registered.
        self.show_manager.render()
        try:
            self.show_manager.window.force_draw()
        except RuntimeError:
            # A draw is already in flight; the scheduled paint will
            # pick up the latest matrix on the next frame.
            pass
        QApplication.processEvents()

    def _on_roi_create_release(self, event):
        """Finalize the in-progress ROI on pointer release."""
        if state_manager.roi_create_mode is None:
            return
        if not self._roi_create_dragging or self._roi_create_initial_pos is None:
            self._roi_create_dragging = False
            self._roi_create_initial_pos = None
            return
        world_pos, _ = self._screen_to_slice_world(event.x, event.y)
        start, axis = self._roi_create_initial_pos
        self._roi_create_initial_pos = None
        self._roi_create_dragging = False
        self._cleanup_roi_preview()
        if world_pos is None:
            self.show_manager.render()
            return
        end = world_pos
        start_arr = np.asarray(start, dtype=np.float64)
        end_arr = np.asarray(end, dtype=np.float64)
        if float(np.linalg.norm(end_arr - start_arr)) <= 0:
            self.show_manager.render()
            return
        self.roi_drawn.emit(
            start_arr,
            end_arr,
            state_manager.roi_create_mode,
        )

    def handle_key_strokes(self, event):
        """Handle key strokes.

        Parameters
        ----------
        event : Event
            The key stroke event.
        """
        if not self._keystrokes_enabled:
            return
        cluster_state_changed = False
        if event.key == "e":
            self.remove_visualization(
                visualization_manager.tractogram_visualizations,
                visualization_type="tractogram",
            )
            visualization_manager.expand_clusters()
            self.add_visualization(
                visualization_manager.tractogram_visualizations,
                visualization_type="tractogram",
            )
            cluster_state_changed = True
        elif event.key == "c":
            self.remove_visualization(
                visualization_manager.tractogram_visualizations,
                visualization_type="tractogram",
            )
            visualization_manager.collapse_clusters()
            self.add_visualization(
                visualization_manager.tractogram_visualizations,
                visualization_type="tractogram",
            )
            cluster_state_changed = True
        elif event.key == "h":
            visualization_manager.hide_clusters()
            cluster_state_changed = True
        elif event.key == "s":
            visualization_manager.show_clusters()
            cluster_state_changed = True
        elif event.key == "a":
            visualization_manager.select_all_clusters()
            cluster_state_changed = True
        elif event.key == "n":
            visualization_manager.select_none_clusters()
            cluster_state_changed = True
        elif event.key == "i":
            visualization_manager.swap_clusters()
            cluster_state_changed = True
        elif event.key == "d":
            self.remove_visualization(
                visualization_manager.tractogram_visualizations,
                visualization_type="tractogram",
            )
            visualization_manager.delete_clusters()
            self.add_visualization(
                visualization_manager.tractogram_visualizations,
                visualization_type="tractogram",
            )
            cluster_state_changed = True
        elif event.key == "x":
            self._keystroke_card.setVisible(not self._keystroke_card.isVisible())

        if cluster_state_changed:
            screen = self.parent()
            refresh = getattr(screen, "_refresh_mesh_projection_if_active", None)
            if callable(refresh):
                refresh()

            self.show_manager.render()
