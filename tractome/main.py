import os

import numpy as np
import trimesh
from dipy.io.image import load_nifti
from dipy.io.streamline import load_tractogram
from fury import actor, window
from fury.colormap import distinguishable_colormap
from fury.geometry import buffer_to_geometry, create_mesh
from fury.material import _create_mesh_material
from fury.utils import get_slices, show_slices
from plyfile import PlyData
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

app = QApplication([])


class Tractome(QMainWindow):
    def __init__(self, *, streamlines=None, mesh=None, mesh_tex=None, t1=None):
        super().__init__()
        self._initUI()
        self._streamlines = streamlines
        self._mesh = mesh
        self._mesh_tex = mesh_tex
        self._t1 = t1
        self._color_generator = distinguishable_colormap()
        self._init_actors()
        # window.update_camera(self.show_manager.screens[0].camera, None, self.scene)

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
        self.scene = window.Scene()

        self.show_manager = window.ShowManager(
            scene=self.scene,
            qt_app=app,
            qt_parent=self,
            window_type="qt",
        )
        self.setWindowTitle("Tractome 2.0")
        self.setGeometry(100, 100, 1200, 700)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel)

        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, 1)

        self.apply_stylesheet()

    def apply_stylesheet(self):
        """
        Applies a global stylesheet to the application to style the widgets.
        """
        self.setStyleSheet("""
            QMainWindow {
                background-color: #FFFFFF;
            }
            QGroupBox {
                background-color: #F0F0F0;
                border: 1px solid #CCCCCC;
                border-radius: 5px;
                margin-top: 1ex; /* space above the title */
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 3px;
                background-color: #F0F0F0;
            }
            QPushButton {
                background-color: #E1E1E1;
                border: 1px solid #ADADAD;
                padding: 5px;
                border-radius: 3px;
                min-height: 20px;
            }
            QPushButton:hover {
                background-color: #D1D1D1;
                border: 1px solid #999999;
            }
            QPushButton:pressed {
                background-color: #C1C1C1;
            }
            QLineEdit {
                border: 1px solid #CCCCCC;
                padding: 5px;
                border-radius: 3px;
            }
            #TitleLabel {
                font-size: 24px;
                font-weight: bold;
                padding-bottom: 10px;
            }
        """)

    def create_left_panel(self):
        """
        Creates the left-side control panel with all its sections and widgets.
        """
        # The left panel uses a QVBoxLayout to stack the different control groups vertically.
        left_panel_widget = QWidget()
        left_layout = QVBoxLayout(left_panel_widget)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(10)

        # Set a fixed width for the control panel
        left_panel_widget.setFixedWidth(250)

        # --- Title ---
        title_label = QLabel("Tractome 2.0")
        title_label.setObjectName("TitleLabel")  # Set object name for styling
        left_layout.addWidget(title_label)

        # --- Section: Data Loading ---
        data_loading_group = QGroupBox("Data Loading")
        data_loading_layout = QHBoxLayout()
        data_loading_layout.addWidget(QPushButton("FIBERS"))
        data_loading_layout.addWidget(QPushButton("IMAGE"))
        data_loading_layout.addWidget(QPushButton("MESH"))
        data_loading_group.setLayout(data_loading_layout)
        left_layout.addWidget(data_loading_group)

        # --- Section: Data Storage ---
        data_storage_group = QGroupBox("Data Storage")
        data_storage_layout = QHBoxLayout()
        data_storage_layout.addWidget(QPushButton("FIBERS"))
        data_storage_layout.addWidget(QPushButton("SESSION"))
        data_storage_group.setLayout(data_storage_layout)
        left_layout.addWidget(data_storage_group)

        # --- Section: Show Data ---
        show_data_group = QGroupBox("Show Data")
        show_data_layout = QGridLayout()
        show_data_layout.addWidget(QPushButton("FIBERS"), 0, 0)
        show_data_layout.addWidget(QPushButton("CLUSTER"), 0, 1)
        show_data_layout.addWidget(QPushButton("IMAGE"), 1, 0)
        show_data_layout.addWidget(QPushButton("MESH"), 1, 1)
        show_data_group.setLayout(show_data_layout)
        left_layout.addWidget(show_data_group)

        # --- Section: Cluster Selection ---
        cluster_selection_group = QGroupBox("Cluster Selection")
        cluster_selection_layout = QHBoxLayout()
        cluster_selection_layout.addWidget(QPushButton("ALL"))
        cluster_selection_layout.addWidget(QPushButton("NULL"))
        cluster_selection_layout.addWidget(QPushButton("SWAP"))
        cluster_selection_group.setLayout(cluster_selection_layout)
        left_layout.addWidget(cluster_selection_group)

        # --- Section: Fibers Selection ---
        fibers_selection_group = QGroupBox("Fibers Selection")
        fibers_selection_layout = QHBoxLayout()
        fibers_selection_layout.addWidget(QLineEdit("1234"))
        fibers_selection_layout.addWidget(QPushButton("APPLY"))
        fibers_selection_group.setLayout(fibers_selection_layout)
        left_layout.addWidget(fibers_selection_group)

        # --- Section: Fibers Selection (More/Less) ---
        fibers_selection_2_group = QGroupBox("Fibers Selection")
        fibers_selection_2_layout = QHBoxLayout()
        fibers_selection_2_layout.addWidget(QPushButton("LESS"))
        fibers_selection_2_layout.addWidget(QPushButton("MORE"))
        fibers_selection_2_group.setLayout(fibers_selection_2_layout)
        left_layout.addWidget(fibers_selection_2_group)

        # --- Section: Session History ---
        session_history_group = QGroupBox("Session History")
        session_history_layout = QHBoxLayout()
        session_history_layout.addWidget(QPushButton("BACK"))
        session_history_layout.addWidget(QPushButton("NEXT"))
        session_history_group.setLayout(session_history_layout)
        left_layout.addWidget(session_history_group)

        # --- Section: Fiber Stats ---
        fiber_stats_group = QGroupBox("Fiber Stats")
        fiber_stats_layout = QFormLayout()
        fiber_stats_layout.addRow("Source fibers:", QLabel("9234"))
        fiber_stats_layout.addRow("Current fibers:", QLabel("5678"))
        fiber_stats_layout.addRow("Selected fibers:", QLabel("3012"))
        fiber_stats_group.setLayout(fiber_stats_layout)
        left_layout.addWidget(fiber_stats_group)

        # Add a spacer to push all content to the top
        left_layout.addStretch(1)

        return left_panel_widget

    def create_right_panel(self):
        """
        Creates the right-side view panel, including the view toggle buttons
        and the main display area.
        """
        # The right panel uses a QVBoxLayout to stack the view toggles above the main view.
        right_panel_widget = QWidget()
        right_layout = QVBoxLayout(right_panel_widget)
        right_layout.setContentsMargins(0, 10, 10, 10)
        right_layout.setSpacing(10)

        # --- View Toggle Buttons ---
        # A QHBoxLayout is used for the buttons, with a spacer to push them to the right.
        toggle_button_layout = QHBoxLayout()
        toggle_button_layout.addStretch(1)  # Pushes buttons to the right
        toggle_button_layout.addWidget(QPushButton("3D VIEW"))
        toggle_button_layout.addWidget(QPushButton("2D VIEW"))
        right_layout.addLayout(toggle_button_layout)

        # --- Main View Area ---
        # A QLabel is used as a placeholder for the actual 2D/3D rendering widget.
        # It's styled with a black background to match the screenshot.

        right_layout.addWidget(self.show_manager.window)

        return right_panel_widget

    def _init_actors(self):
        if self._streamlines is not None:
            colors = np.zeros((len(self._streamlines), 3))
            print(len(self._streamlines))
            colors = np.tile(next(self._color_generator), (len(self._streamlines), 1))
            self._streamlines_actor = actor.line(self._streamlines, colors=colors)
            self._streamlines_actor.local.position = (0, 0, 0)
            self.scene.add(self._streamlines_actor)

        if self._mesh is not None:
            vertices = self._mesh.vertices * 10e5
            faces = self._mesh.faces

            self._mesh_actor = actor.surface(
                vertices,
                faces,
                material="basic",
                texture=self._mesh_tex,
                texture_coords=self._mesh.visual.uv,
            )
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


def tractome(tractogram_path=None, mesh_path=None, t1_path=None, mesh_tex_path=None):
    streamlines = None
    mesh = None
    t1 = None
    if tractogram_path is not None:
        sft = load_tractogram(tractogram_path, "same", bbox_valid_check=False)
        streamlines = sft.streamlines
    if mesh_path is not None:
        mesh = trimesh.load(mesh_path, process=False)
    if t1_path is not None:
        t1, _ = load_nifti(t1_path)
    tractome_app = Tractome(
        streamlines=streamlines, mesh=mesh, mesh_tex=mesh_tex_path, t1=t1
    )
    tractome_app.show()
    tractome_app.show_manager.start()


if __name__ == "__main__":
    tractogram_path = os.path.expanduser(
        "~/.dipy/bundle_atlas_hcp842/Atlas_80_Bundles/bundles/AC.trk"
    )
    mesh_path = os.path.expanduser(
        "/Volumes/PortableSSD/GRG/Tractome/sub-01_epo-01.obj"
    )
    mesh_tex_path = os.path.expanduser(
        "/Volumes/PortableSSD/GRG/Tractome/sub-01_epo-01_ref-01_mesh.jpg"
    )
    t1_path = os.path.expanduser(
        "~/.dipy/mni_template/mni_icbm152_t1_tal_nlin_asym_09a.nii"
    )
    tractome(
        tractogram_path,
        mesh_path=mesh_path,
        t1_path=None,
        mesh_tex_path=mesh_tex_path,
    )
