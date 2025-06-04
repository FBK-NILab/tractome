import os

import numpy as np
from dipy.io.image import load_nifti
from dipy.io.streamline import load_tractogram
from fury import actor, window
from fury.colormap import distinguishable_colormap
from fury.geometry import buffer_to_geometry, create_mesh
from fury.lib import QtWidgets
from fury.material import _create_mesh_material
from fury.utils import get_slices, show_slices
from plyfile import PlyData

app = QtWidgets.QApplication([])


class Tractome(QtWidgets.QWidget):
    def __init__(self, *, streamlines=None, mesh=None, t1=None):
        super().__init__(None)
        self._initUI()
        self._streamlines = streamlines
        self._mesh = mesh
        self._t1 = t1
        self._color_generator = distinguishable_colormap()
        self._init_actors()
        window.update_camera(self.show_manager.screens[0].camera, None, self.scene)

    def _on_mesh_button_click(self):
        if self._mesh_button.text() == "Show Mesh":
            self.scene.add(self._mesh_actor)
            self._mesh_button.setText("Hide Mesh")
        else:
            self.scene.remove(self._mesh_actor)
            self._mesh_button.setText("Show Mesh")
        self.show_manager.render()

    def _on_key_press(self, event):
        position = get_slices(self._t1_actor)
        if event.key == "ArrowUp":
            position += 1
        elif event.key == "ArrowDown":
            position -= 1

        position = np.maximum(np.zeros((3,)), position)
        position = np.minimum(np.asarray(self._t1.shape), position)
        show_slices(self._t1_actor, position)
        self.show_manager.render()

    def _on_streamlines_button_click(self):
        if self._streamlines_button.text() == "Show Streamlines":
            self.scene.add(self._streamlines_actor)
            self._streamlines_button.setText("Hide Streamlines")
        else:
            self.scene.remove(self._streamlines_actor)
            self._streamlines_button.setText("Show Streamlines")
        self.show_manager.render()

    def _initUI(self):
        self.setWindowTitle("Tractome 2.0")
        self.resize(800, 600)
        self.scene = window.Scene()

        self._mesh_button = QtWidgets.QPushButton("Hide Mesh", self)
        self._mesh_button.clicked.connect(self._on_mesh_button_click)

        self._streamlines_button = QtWidgets.QPushButton("Hide Streamlines", self)
        self._streamlines_button.clicked.connect(self._on_streamlines_button_click)

        self.show_manager = window.ShowManager(
            scene=self.scene,
            qt_app=app,
            qt_parent=self,
            window_type="qt",
        )

        self.show_manager.renderer.add_event_handler(self._on_key_press, "key_down")

        layout = QtWidgets.QHBoxLayout()
        control_container = QtWidgets.QWidget()
        control_container.setFixedWidth(200)
        control_layout = QtWidgets.QVBoxLayout(control_container)
        control_layout.addWidget(self._mesh_button)
        control_layout.addWidget(self._streamlines_button)
        layout.addWidget(control_container)
        layout.addWidget(self.show_manager.window)
        self.setLayout(layout)

    def _init_actors(self):
        if self._streamlines is not None:
            colors = np.zeros((len(self._streamlines), 3))
            colors = np.tile(next(self._color_generator), (len(self._streamlines), 1))
            self._streamlines_actor = actor.line(self._streamlines, colors=colors)
            self._streamlines_actor.local.position = (0, 0, 0)
            self.scene.add(self._streamlines_actor)

        if self._mesh is not None:
            vertices = self._mesh["vertex"].data
            pos = np.stack((vertices["x"], vertices["y"], vertices["z"]), axis=-1)
            pos *= 1000
            faces = self._mesh["face"].data
            indices = faces["vertex_indices"]
            indices = np.vstack(indices)
            colors = np.zeros((len(pos), 3))
            colors = np.tile(next(self._color_generator), (len(pos), 1))
            geo = buffer_to_geometry(
                positions=pos, indices=indices, colors=colors.astype(np.float32)
            )
            mat = _create_mesh_material(mode="vertex")
            self._mesh_actor = create_mesh(geo, mat)
            self._mesh_actor.local.position = (0, 0, 0)
            self.scene.add(self._mesh_actor)

        if self._t1 is not None:
            self._t1_actor = actor.slicer(self._t1)
            self.scene.add(self._t1_actor)
            self._t1_actor.local.position = (0, 0, 0)

    @property
    def streamlines(self):
        return self._streamlines

    @streamlines.setter
    def streamlines(self, streamlines):
        self._streamlines = streamlines
        if self._streamlines_actor in self.scene.children:
            self.scene.remove(self._streamlines_actor)
        self._streamlines_actor = actor.line(self._streamlines)
        self.scene.add(self._streamlines_actor)
        self.show_manager.render()


def tractome(tractogram_path=None, mesh_path=None, t1_path=None):
    streamlines = None
    plydata = None
    t1 = None
    if tractogram_path is not None:
        sft = load_tractogram(tractogram_path, "same", bbox_valid_check=False)
        streamlines = sft.streamlines
    if mesh_path is not None:
        plydata = PlyData.read(mesh_path)
    if t1_path is not None:
        t1, _ = load_nifti(t1_path)
    tractome_app = Tractome(streamlines=streamlines, mesh=plydata, t1=t1)
    tractome_app.show_manager.start()


if __name__ == "__main__":
    tractogram_path = os.path.expanduser(
        "~/.dipy/bundle_atlas_hcp842/Atlas_80_Bundles/bundles/AC.trk"
    )
    mesh_path = os.path.expanduser("~/Downloads/sub-01_epo-01.ply")
    t1_path = os.path.expanduser(
        "~/.dipy/mni_template/mni_icbm152_t1_tal_nlin_asym_09a.nii"
    )
    tractome(tractogram_path, mesh_path=mesh_path, t1_path=t1_path)
