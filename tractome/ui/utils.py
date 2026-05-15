import logging
from pathlib import Path
import re

from PySide6.QtWidgets import QFileDialog

ASSETS_PATH = Path(__file__).resolve().parent.parent / "assets"


def load_style_sheet():
    """Load the stylesheet for the app.

    Returns
    -------
    str
        String value of the stylesheet.
    """
    style_path = ASSETS_PATH / "style.qss"
    with style_path.open("r", encoding="utf-8") as f:
        style_sheet = f.read()

    # Qt resolves url(...) entries from the process cwd, not this qss file.
    # Rewrite relative asset URLs to absolute paths so styles load reliably.
    def _resolve_qss_url(match):
        raw_path = match.group("path").strip().strip("\"'")
        if (
            not raw_path
            or raw_path.startswith(("file:", "qrc:", ":", "http://", "https://"))
            or raw_path.startswith("/")
            or re.match(r"^[A-Za-z]:[/\\\\]", raw_path)
        ):
            return match.group(0)

        abs_path = (ASSETS_PATH / raw_path).resolve().as_posix()
        return f'url("{abs_path}")'

    return re.sub(r"url\((?P<path>[^)]+)\)", _resolve_qss_url, style_sheet)


def open_file_dialog(
    *, parent=None, title="Select a file", file_filter="All Files (*.*)"
):
    """Open a file dialog and return the selected file path.

    Parameters
    ----------
    title : str, optional
        Title of the file dialog, by default "Select a file"
    file_filter : str, optional
        Filter for the file types, by default "All Files (*.*)"

    Returns
    -------
    str
        Path to the selected file.
    """
    file_path, _ = QFileDialog.getOpenFileName(parent, title, "", file_filter)
    if file_path:
        logging.info(f"Selected file: {file_path}")
    else:
        logging.info("No file selected.")
    return file_path


def save_file_dialog(
    *,
    parent=None,
    title="Save file",
    file_filter="All Files (*.*)",
    default_name="",
):
    """Open a save dialog and return the chosen file path."""
    file_path, _ = QFileDialog.getSaveFileName(parent, title, default_name, file_filter)
    if file_path:
        logging.info(f"Save target: {file_path}")
    return file_path
