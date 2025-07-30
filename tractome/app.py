from PySide6.QtWidgets import QApplication, QMainWindow
import numpy as np

from fury import window
from fury.utils import set_group_visibility, show_slices
from tractome.io import read_mesh, read_nifti, read_tractogram
from tractome.ui import STYLE_SHEET, create_slice_sliders, create_ui
from tractome.viz import create_image_slicer, create_mesh, create_tractogram

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
        self._3D_actors = {"t1": None, "tractogram": None, "mesh": None}
        self._2D_actors = []
        self._init_UI()
        self._init_actors()
        window.update_camera(self.show_manager.screens[0].camera, None, self.scene)

    def _init_UI(self):
        """Initialize the user interface."""
        self.resize(800, 800)
        self.setWindowTitle("Tractome 2.0")
        self.scene = window.Scene()

        self.show_manager = window.ShowManager(
            scene=self.scene,
            qt_app=app,
            qt_parent=self,
            window_type="qt",
        )

        main_widget, left_panel, right_panel = create_ui(self.show_manager.window)
        self.setCentralWidget(main_widget)
        self.left_panel = left_panel
        self.right_panel = right_panel
        self.setStyleSheet(STYLE_SHEET)

    def _init_actors(self):
        """Initialize the actors for the scene."""
        if self.tractogram:
            sft = read_tractogram(self.tractogram)
            tractogram_actor = create_tractogram(sft)
            self.scene.add(tractogram_actor)

        if self.mesh:
            mesh_obj, texture = read_mesh(self.mesh, texture=self.mesh_texture)
            mesh_actor = create_mesh(mesh_obj, texture=texture)
            self.scene.add(mesh_actor)

        if self.t1:
            nifti_img, affine = read_nifti(self.t1)
            image_slicer = create_image_slicer(nifti_img, affine=affine)
            self.scene.add(image_slicer)
            min_vals, max_vals = image_slicer.get_bounding_box()
            slider_widget, sliders, checkboxes = create_slice_sliders(
                min_vals=np.asarray(min_vals, dtype=np.int32),
                max_vals=np.asarray(max_vals, dtype=np.int32),
            )
            self.left_panel.layout().addWidget(slider_widget)
            self._x_slider, self._y_slider, self._z_slider = sliders

            for slider in sliders:
                slider.valueChanged.connect(self.update_slices)

            self._x_checkbox, self._y_checkbox, self._z_checkbox = checkboxes

            for checkbox in checkboxes:
                checkbox.stateChanged.connect(self.update_slice_visibility)

            self._3D_actors["t1"] = image_slicer
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
        self.show_manager.render()

    def update_slice_visibility(self, _value):
        """Update the visibility of the slices based on checkbox states. Callback
        for the checkbox changes.

        Parameters
        ----------
        _value : bool
            The current checked state of the checkbox.
        """
        checkbox_states = self.get_current_checkbox_states()
        set_group_visibility(self._3D_actors["t1"], checkbox_states)
        self.show_manager.render()
