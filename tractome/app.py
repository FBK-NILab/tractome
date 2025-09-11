from PySide6.QtWidgets import QApplication, QMainWindow
import numpy as np
from pygfx import OrthographicCamera, TrackballController

from fury import window
from fury.lib import DirectionalLight, PerspectiveCamera
from fury.utils import set_group_visibility, show_slices
from tractome.compute import mkbm_clustering
from tractome.io import read_mesh, read_nifti, read_tractogram
from tractome.ui import (
    STYLE_SHEET,
    create_clusters_slider,
    create_slice_sliders,
    create_ui,
)
from tractome.viz import (
    create_image_slicer,
    create_mesh,
    create_streamlines,
    create_streamlines_projection,
    create_streamtube,
)

app = QApplication([])


class Tractome(QMainWindow):
    def __init__(self, tractogram=None, mesh=None, mesh_texture=None, t1=None):
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
        """
        super().__init__()
        self.tractogram = tractogram
        self.mesh = mesh
        self.mesh_texture = mesh_texture
        self.t1 = t1
        self._mode = "3D"
        self._3D_actors = {"t1": None, "tractogram": None, "mesh": None}
        self._2D_actors = {"t1": None, "tractogram": None, "mesh": None}
        self._sft = None
        self._clusters = None
        self._cluster_reps = {}
        self._streamline_bundles = []
        self._selected_clusters = set()
        self._streamline_projections = set()
        self._init_UI()
        self._init_actors()

    def _init_UI(self):
        """Initialize the user interface."""
        self.resize(800, 800)
        self.setWindowTitle("Tractome 2.0")

        self._3D_scene = window.Scene()
        self._2D_scene = window.Scene()

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
            blend_mode="weighted_plus",
        )

        # TODO: Remove long press event handler for Qt
        # This is a temporary workaround for the long press issue in Qt
        self.show_manager.renderer.remove_event_handler(
            self.show_manager._set_key_long_press_event, "key_up", "key_down"
        )

        self._3D_controller = TrackballController(
            self._3D_camera, register_events=self.show_manager.renderer
        )
        self._2D_controller = OrbitController(
            self._2D_camera, register_events=self.show_manager.renderer
        )
        self._2D_controller.enabled = False

        (
            main_widget,
            self.left_panel,
            self.right_panel,
            self._toggle_3d,
            self._toggle_2d,
        ) = create_ui(self.show_manager.window)
        self._toggle_3d.clicked.connect(self.toggle_3D_mode)
        self._toggle_2d.clicked.connect(self.toggle_2D_mode)
        self.setCentralWidget(main_widget)
        self.setStyleSheet(STYLE_SHEET)

        # Toggle Control Widgets
        self._slice_widget = None
        self._3D_check_box_values = None
        self._2D_radio_buttons_values = None

    def _init_actors(self):
        """Initialize the actors for the scene."""
        if self.tractogram:
            self._sft = read_tractogram(self.tractogram, reference=self.t1)
            if (
                self._sft.data_per_streamline is None
                or "dismatrix" not in self._sft.data_per_streamline
            ):
                tractogram = create_streamlines(self._sft.streamlines, color=(0, 1, 0))
                self._3D_scene.add(tractogram)
            else:
                self.perform_clustering(100)
                self.show_manager.renderer.add_event_handler(
                    self.handle_key_strokes, "key_down"
                )
                self._cluster_widget, self._cluster_slider, _ = create_clusters_slider(
                    default_value=100
                )
                self.left_panel.layout().addWidget(self._cluster_widget)
                self._cluster_slider.valueChanged.connect(
                    lambda: self.perform_clustering(self._cluster_slider.value())
                )

        if self.mesh:
            print("Loading mesh...")
            mesh_obj, texture = read_mesh(self.mesh, texture=self.mesh_texture)
            mesh_actor = create_mesh(mesh_obj, texture=texture)
            self._3D_scene.add(mesh_actor)

        if self.t1:
            nifti_img, affine = read_nifti(self.t1)

            image_slicer = create_image_slicer(nifti_img, affine=affine)
            self._3D_scene.add(image_slicer)

            image_slice = create_image_slicer(nifti_img, affine=affine)
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

        self._3D_camera.show_object(self._3D_scene, (0, 0, 1))
        self.show_manager.start()

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
        [show_slices(projection, slices) for projection in self._streamline_projections]
        self.show_manager.render()

    def update_slice_visibility(self, _value):
        """Update the visibility of the slices based on checkbox states. Callback
        for the checkbox changes.

        Parameters
        ----------
        _value : bool
            The current checked state of the checkbox.
        """
        if self._mode == "3D":
            checkbox_states = self.get_current_checkbox_states()
            set_group_visibility(self._3D_actors["t1"], checkbox_states)
        elif self._mode == "2D":
            radio_states = self.get_current_checkbox_states()
            # This is required to reset the camera orientation.
            self._2D_camera.show_object(self._2D_actors["t1"], (0, 0, -1))
            self._2D_camera.show_object(
                self._2D_actors["t1"], tuple(np.asarray(radio_states, dtype=int))
            )
            self._3D_camera.show_object(self._3D_actors["t1"], (0, 0, -1))
            self._3D_camera.show_object(
                self._3D_actors["t1"], tuple(np.asarray(radio_states, dtype=int))
            )
            set_group_visibility(self._2D_actors["t1"], radio_states)
            [
                set_group_visibility(proj, radio_states)
                for proj in self._streamline_projections
            ]
        self.show_manager.render()

    def perform_clustering(self, value):
        self._selected_clusters.clear()
        self._collapse_streamline_bundles()
        self._clusters = mkbm_clustering(
            self._sft.data_per_streamline["dismatrix"],
            n_clusters=value,
            streamline_ids=np.indices((len(self._sft.streamlines),))[0],
        )
        self._3D_scene.remove(*self._cluster_reps.values())
        self._cluster_reps = create_streamtube(self._clusters, self._sft.streamlines)
        for cluster in self._cluster_reps.values():
            cluster.add_event_handler(self.toggle_cluster_selection, "pointer_down")
            self._3D_scene.add(cluster)
        self.show_manager.render()

    def toggle_cluster_selection(self, event):
        """Toggle the selection state of a cluster.

        Parameters
        ----------
        event : Event
            The click event.
        """
        cluster = event.target
        if cluster in self._selected_clusters:
            self._selected_clusters.remove(cluster)
        else:
            self._selected_clusters.add(cluster)

    def handle_key_strokes(self, event):
        if event.key == "e":
            for cluster in self._selected_clusters:
                if cluster in self._3D_scene.children:
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
        elif event.key == "c":
            self._collapse_streamline_bundles()
        elif event.key == "h":
            for cluster in self._cluster_reps.values():
                if (
                    cluster not in self._selected_clusters
                    and cluster in self._3D_scene.children
                ):
                    self._3D_scene.remove(cluster)

        elif event.key == "s":
            for cluster in self._cluster_reps.values():
                if (
                    cluster not in self._3D_scene.children
                    and cluster not in self._selected_clusters
                ):
                    self._3D_scene.add(cluster)

        self.show_manager.render()

    def _collapse_streamline_bundles(self):
        """Collapse all streamline bundles."""
        for bundle in self._streamline_bundles:
            self._3D_scene.remove(bundle)
            self._3D_scene.add(self._cluster_reps[bundle.rep])
        self._streamline_bundles = []

    def _create_streamlines_projection(self):
        self._2D_scene.remove(*self._streamline_projections)
        self._streamline_projections.clear()
        for cluster in self._selected_clusters:
            streamlines = [
                np.asarray(self._sft.streamlines[line])
                for line in self._clusters[cluster.rep]
            ]
            projection = create_streamlines_projection(
                streamlines=streamlines,
                colors=cluster.geometry.colors.data[0],
                slice_values=self.get_current_slider_position(),
            )
            self._streamline_projections.add(projection)
        self._2D_scene.add(*self._streamline_projections)

    def toggle_3D_mode(self):
        """Toggle to 3D mode."""
        if self._mode != "3D":
            self._mode = "3D"
            self.show_manager.screens[0].scene = self._3D_scene
            self.show_manager.screens[0].camera = self._3D_camera
            self.show_manager.screens[0].controller = self._3D_controller
            self._3D_controller.enabled = True
            self._2D_controller.enabled = False
            # self._3D_camera.show_object(self._3D_scene, (0, 0, 1))
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

    def toggle_2D_mode(self):
        """Toggle to 2D mode."""
        if self._mode != "2D":
            self._mode = "2D"
            self.show_manager.screens[0].scene = self._2D_scene
            self.show_manager.screens[0].camera = self._2D_camera
            self.show_manager.screens[0].controller = self._2D_controller
            self._3D_controller.enabled = False
            self._2D_controller.enabled = True
            self._2D_camera.show_object(self._2D_scene, (0, 0, -1))
            self._create_streamlines_projection()

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
                self._2D_scene, tuple(np.asarray(radio_states, dtype=int))
            )
            self.update_slice_visibility(None)
            self.show_manager.render()
