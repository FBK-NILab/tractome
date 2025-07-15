from fury import window
from PySide6.QtWidgets import QApplication, QMainWindow

from tractome.io import read_mesh, read_nifti, read_tractogram
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

        # TODO: Create methods to update the ui elements

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
        self.show_manager.start()
