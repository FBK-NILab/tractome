"""UI module for the tractome app."""

__all__ = [
    "ASSETS_PATH",
    "IMAGES_PATH",
    "ICONS_PATH",
    "ClustersWidget",
    "LeftSectionWidget",
    "RightSectionWidget",
    "CenterSectionWidget",
    "StartScreen",
    "InteractionScreen",
    "ViewModeWidget",
    "load_style_sheet",
    "open_file_dialog",
]

from tractome.ui._control_section import (
    ClustersWidget,
    LeftSectionWidget,
    ViewModeWidget,
)
from tractome.ui._input_section import RightSectionWidget
from tractome.ui._paths import ASSETS_PATH, ICONS_PATH, IMAGES_PATH
from tractome.ui._views import InteractionScreen, StartScreen
from tractome.ui._visualization_section import CenterSectionWidget
from tractome.ui.utils import load_style_sheet, open_file_dialog
