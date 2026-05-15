from PySide6.QtWidgets import QApplication, QMainWindow, QStackedWidget

from tractome.io import get_file_extension
from tractome.mem import input_manager, state_manager, visualization_manager
from tractome.ui import InteractionScreen, StartScreen, load_style_sheet

app = QApplication.instance() or QApplication([])


class Tractome(QMainWindow):
    """Tractome is a tool for analyzing and visualizing brain tractography data.

    It provides a pipeline for processing tractograms, meshes, and other related data,
    as well as a command-line interface for running the
    pipeline and computing dissimilarity matrices."""

    def __init__(
        self,
        tractogram=None,
        t1=None,
        mesh=None,
        mesh_texture=None,
        roi=None,
        parcel=None,
    ):
        """Initialize the Tractome pipeline.

        Parameters
        ----------
        tractogram : str, optional
            Path to the tractogram file
        t1 : str, optional
            Path to the T1-weighted image file
        mesh : str, optional
            Path to the mesh file
        mesh_texture : str, optional
            Path to the mesh texture file
        roi : list[str], optional
            List of paths to ROI files
        parcel : str, optional
            Path to a parcel CSV file
        """
        super().__init__()
        self._initialize_input_manager(tractogram, t1, mesh, mesh_texture, roi, parcel)
        self._initialize_window()

    def _initialize_input_manager(
        self, tractogram, t1, mesh, mesh_texture, roi, parcel
    ):
        """Initialize the input manager with pre-load files.

        Parameters
        ----------
        tractogram : str
            Path of tractogram.
        t1 : str
            Path of T1 image.
        mesh : str
            Path of surface mesh.
        mesh_texture : str
            Path of image texture for the mesh.
        roi : str
            Path of roi to showcase.
        parcel : str
            Path of parcel to showcase.
        """
        if tractogram is not None:
            input_manager.add_tractogram(tractogram)
        if t1 is not None:
            input_manager.add_t1(t1)
        if mesh is not None and mesh_texture is not None:
            input_manager.add_mesh(mesh, mesh_texture)
        if roi is not None:
            for roi_path in roi:
                input_manager.add_roi(roi_path)
        if parcel is not None:
            input_manager.add_parcel(parcel)

    def _completed_start_screen(self, file_path):
        """Handle the completion of the start screen.

        Parameters
        ----------
        file_path : str
            Path of the uploaded file.
        """
        if file_path is not None:
            self._file_uploaded(file_path)
        self._stack.setCurrentIndex(1)
        self._visualize_inputs()

    def _file_uploaded(self, file_path):
        """Handle the file uploaded event.

        Parameters
        ----------
        file_path : str
            Path of the uploaded file.
        """

        ext = get_file_extension(file_path)

        if ext in (".trx", ".trk"):
            input_manager.add_tractogram(file_path)

    def _initialize_window(self):
        """Initialize the window"""
        self.setWindowTitle("Tractome")
        self.resize(1200, 800)
        style_sheet = load_style_sheet()
        self.setStyleSheet(style_sheet)

        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        self._start_screen = StartScreen(on_uploading_done=self._completed_start_screen)
        self._stack.addWidget(self._start_screen)

        self._interaction_screen = InteractionScreen()
        self._stack.addWidget(self._interaction_screen)

        if input_manager.has_input:
            self._completed_start_screen(None)

    def _visualize_inputs(self):
        """Visualize the inputs in the interaction screen."""
        t1_visualization = visualization_manager.visualize_t1()
        if t1_visualization is not None:
            self._interaction_screen.add_visualization(
                t1_visualization, visualization_type="t1"
            )
        tractogram_visualization = visualization_manager.visualize_tractogram()
        if tractogram_visualization is not None:
            self._interaction_screen.add_visualization(
                tractogram_visualization, visualization_type="tractogram"
            )
        mesh_visualization = visualization_manager.visualize_mesh()
        if mesh_visualization is not None:
            self._interaction_screen.add_visualization(
                mesh_visualization, visualization_type="mesh"
            )
            self._interaction_screen._right_section.mesh_input_widget.sync_mesh_visibility_button()
        parcel_visualization = visualization_manager.visualize_parcel()
        if parcel_visualization is not None:
            self._interaction_screen.add_visualization(
                parcel_visualization, visualization_type="parcel"
            )
            self._interaction_screen._right_section.parcel_input_widget.sync_parcel_visibility_button()
        roi_visualization = visualization_manager.visualize_rois()
        if roi_visualization:
            self._interaction_screen.add_visualization(
                roi_visualization, visualization_type="roi"
            )
            self._interaction_screen._left_section.roi_input_widget.refresh_rois()

        if (
            visualization_manager.apply_roi_filter()
            and tractogram_visualization is not None
        ):
            self._interaction_screen.remove_visualization(
                tractogram_visualization, visualization_type="tractogram"
            )
            tractogram_visualization = visualization_manager.visualize_tractogram(
                nb_clusters=state_manager.get_latest_state().nb_clusters,
            )
            if tractogram_visualization is not None:
                self._interaction_screen.add_visualization(
                    tractogram_visualization, visualization_type="tractogram"
                )

    def start(self):
        """Show the main window and start the FURY/Qt loop."""
        self.show()
        self._interaction_screen._center_section.show_manager.start()


if __name__ == "__main__":
    tractome = Tractome(tractogram="computed.trx")
    tractome.start()
