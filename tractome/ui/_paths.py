"""Shared UI asset paths."""

from pathlib import Path

UI_PACKAGE_PATH = Path(__file__).resolve().parent
ASSETS_PATH = UI_PACKAGE_PATH.parent / "assets"
IMAGES_PATH = ASSETS_PATH / "images"
ICONS_PATH = ASSETS_PATH / "icons"
