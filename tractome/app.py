from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QMainWindow,
    QMessageBox,
    QStackedWidget,
)
import click

from tractome.io import get_file_extension
from tractome.mem import (
    input_manager,
    recovery_manager,
    state_manager,
    visualization_manager,
)
from tractome.ui import (
    EmbeddingSelectionDialog,
    InteractionScreen,
    StartScreen,
    load_style_sheet,
)
from tractome.ui.utils import ASSETS_PATH

app = QApplication.instance() or QApplication([])
APP_ICON_PATH = ASSETS_PATH / "images" / "logo.png"
app.setWindowIcon(QIcon(str(APP_ICON_PATH)))


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
        self.setWindowIcon(QIcon(str(APP_ICON_PATH)))
        self.resize(1200, 800)
        style_sheet = load_style_sheet()
        self.setStyleSheet(style_sheet)

        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)
        self._retired_interaction_screens = []

        self._start_screen = StartScreen(on_uploading_done=self._completed_start_screen)
        self._stack.addWidget(self._start_screen)

        self._interaction_screen = InteractionScreen()
        self._interaction_screen.change_tractogram_requested.connect(
            self._confirm_change_tractogram
        )
        self._stack.addWidget(self._interaction_screen)

        if input_manager.has_input:
            self._completed_start_screen(None)

    def _resolve_embedding_selection(self):
        """Ensure an embedding is chosen for the current tractogram.

        Embeddings are recognized generically: any per-streamline vector on
        the tractogram counts, regardless of its name/type. When several are
        present the user picks one via a radio-button dialog; a single one is
        selected automatically; when none are present the user is offered to
        generate a dissimilarity embedding on the fly.

        Returns
        -------
        bool
            True if an embedding is selected and clustering can proceed,
            False if the user declined to generate one when none existed.
        """
        if not input_manager.has_tractogram:
            return False
        if input_manager.selected_embedding is not None:
            return True

        embedding_keys = input_manager.get_embedding_keys()

        if len(embedding_keys) == 1:
            input_manager.set_selected_embedding(embedding_keys[0])
            return True

        if len(embedding_keys) >= 2:
            dialog = EmbeddingSelectionDialog(embedding_keys, self)
            if dialog.exec() == QDialog.Accepted and dialog.selected_embedding:
                input_manager.set_selected_embedding(dialog.selected_embedding)
            else:
                input_manager.set_selected_embedding(embedding_keys[0])
            return True

        result = QMessageBox.question(
            self,
            "No embeddings found",
            "This tractogram has no embeddings.\n"
            "Generate a dissimilarity embedding now?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if result != QMessageBox.Yes:
            return False

        name = visualization_manager.generate_dissimilarity_embedding()
        if name is None:
            return False
        input_manager.set_selected_embedding(name)
        return True

    def _visualize_inputs(self):
        """Visualize the inputs in the interaction screen."""
        t1_visualization = visualization_manager.visualize_t1()
        if t1_visualization is not None:
            self._interaction_screen.add_visualization(
                t1_visualization, visualization_type="t1"
            )
        if input_manager.has_tractogram:
            self._resolve_embedding_selection()
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

    def _confirm_change_tractogram(self):
        """Ask before resetting the app to choose a different tractogram."""
        result = QMessageBox.question(
            self,
            "Change tractogram",
            "Changing the tractogram will reset the application. Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if result == QMessageBox.Yes:
            self._reset_to_start_screen()

    def _reset_to_start_screen(self):
        """Reset managers and return to the tractogram upload screen."""
        input_manager.reset()
        state_manager.reset()
        recovery_manager.reset()
        visualization_manager.reset()

        old_interaction_screen = self._interaction_screen
        self._stack.removeWidget(old_interaction_screen)
        self._retired_interaction_screens.append(old_interaction_screen)

        self._interaction_screen = InteractionScreen()
        self._interaction_screen.change_tractogram_requested.connect(
            self._confirm_change_tractogram
        )
        self._stack.addWidget(self._interaction_screen)
        self._stack.setCurrentWidget(self._start_screen)

    def start(self):
        """Show the main window and start the FURY/Qt loop."""
        self.show()
        self._interaction_screen._center_section.show_manager.start()


@click.command(name="tractome")
@click.option(
    "--tractogram", type=click.Path(exists=True), help="Path to the tractogram file."
)
@click.option("--mesh", type=click.Path(exists=True), help="Path to the mesh file.")
@click.option(
    "--mesh_texture",
    type=click.Path(exists=True),
    help="Path to the mesh texture file.",
)
@click.option(
    "--t1", type=click.Path(exists=True), help="Path to the T1-weighted image file."
)
@click.option(
    "--roi",
    type=click.Path(exists=True),
    multiple=True,
    help="Path to an ROI file. Use multiple times for multiple ROIs.",
)
@click.option(
    "--parcel",
    type=click.Path(exists=True),
    help=("Path to a parcel CSV file."),
)
def tractome(
    tractogram=None, mesh=None, mesh_texture=None, t1=None, roi=(), parcel=None
):
    """Run the Tractome pipeline.

    Parameters
    ----------
    tractogram : str, optional
        Path to the tractogram file
    mesh : str, optional
        Path to the mesh file
    mesh_texture : str, optional
        Path to the mesh texture file
    t1 : str, optional
        Path to the T1-weighted image file
    roi : tuple[str], optional
        One or more paths to ROI files
    parcel : str, optional
        Path to a parcel CSV file
    """
    tractome = Tractome(tractogram, t1, mesh, mesh_texture, roi, parcel)
    tractome.start()


def main():
    """Entry point for the Tractome application."""
    tractome = Tractome()
    tractome.start()
