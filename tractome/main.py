import os

import numpy as np
from dipy.io.streamline import load_tractogram
from fury import actor, window
from fury.colormap import distinguishable_colormap
from fury.lib import QtWidgets

app = QtWidgets.QApplication([])


class Tractome(QtWidgets.QWidget):
    def __init__(self, *, streamlines=None):
        super().__init__(None)
        self.initUI()
        self._streamlines = streamlines
        self._color_generator = distinguishable_colormap()
        colors = np.zeros((len(self._streamlines), 3))
        colors = np.tile(next(self._color_generator), (len(self._streamlines), 1))

        self._streamlines_actor = actor.line(
            self._streamlines, colors=colors, line_width=2
        )
        self.scene.add(self._streamlines_actor)
        window.update_camera(
            self.show_manager.screens[0].camera, None, self._streamlines_actor
        )

    def initUI(self):
        self.setWindowTitle("Tractome 2.0")
        self.resize(800, 600)
        self.scene = window.Scene()
        self.show_manager = window.ShowManager(
            scene=self.scene, qt_app=app, qt_parent=self
        )

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


def tractome(tractogram_path):
    sft = load_tractogram(tractogram_path, "same", bbox_valid_check=False)
    tractome_app = Tractome(streamlines=sft.streamlines)
    tractome_app.show_manager.start()


if __name__ == "__main__":
    tractogram_path = os.path.expanduser(
        "~/.dipy/bundle_atlas_hcp842/Atlas_80_Bundles/bundles/AC.trk"
    )
    tractome(tractogram_path)
