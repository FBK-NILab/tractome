from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QAction, QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
)

from tractome.mem import input_manager, state_manager, visualization_manager
from tractome.ui._input_section import RoiInputWidget
from tractome.ui._paths import ICONS_PATH


class ViewModeWidget(QFrame):
    """Bundle Identification panel with a 3D / 2D scene toggle."""

    view_mode_changed = Signal(str)

    def __init__(self, *, parent=None):
        """Build the view-mode toggle widget.

        Parameters
        ----------
        parent : QWidget, optional
            The parent widget.
        """
        super().__init__(parent)
        self.setObjectName("viewModeWidget")

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(8, 8, 8, 8)
        self.main_layout.setSpacing(8)

        self.title = QLabel("Bundle identification")
        self.title.setObjectName("viewModeTitle")
        self.main_layout.addWidget(self.title)

        toggle_row = QHBoxLayout()
        toggle_row.setSpacing(0)
        toggle_row.setContentsMargins(0, 0, 0, 0)

        self._button_group = QButtonGroup(self)
        self._button_group.setExclusive(True)

        self.btn_3d = QPushButton("3D")
        self.btn_3d.setObjectName("viewModeButton3D")
        self.btn_3d.setProperty("class", "viewModeButton")
        self.btn_3d.setCheckable(True)
        self.btn_3d.setChecked(True)
        self.btn_3d.setCursor(Qt.PointingHandCursor)
        self.btn_3d.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._button_group.addButton(self.btn_3d)
        toggle_row.addWidget(self.btn_3d)

        self.btn_2d = QPushButton("2D")
        self.btn_2d.setObjectName("viewModeButton2D")
        self.btn_2d.setProperty("class", "viewModeButton")
        self.btn_2d.setCheckable(True)
        self.btn_2d.setCursor(Qt.PointingHandCursor)
        self.btn_2d.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._button_group.addButton(self.btn_2d)
        toggle_row.addWidget(self.btn_2d)

        self.main_layout.addLayout(toggle_row)

        self.btn_3d.clicked.connect(lambda: self._on_mode_clicked("3D"))
        self.btn_2d.clicked.connect(lambda: self._on_mode_clicked("2D"))

    def _on_mode_clicked(self, mode):
        """Emit ``view_mode_changed`` only when the active mode actually flips.

        Parameters
        ----------
        mode : str
            Either ``"3D"`` or ``"2D"``.
        """
        if mode == "3D" and not self.btn_3d.isChecked():
            self.btn_3d.setChecked(True)
        if mode == "2D" and not self.btn_2d.isChecked():
            self.btn_2d.setChecked(True)
        self.view_mode_changed.emit(mode)

    def set_mode(self, mode):
        """Update the toggle's checked state without emitting a signal.

        Parameters
        ----------
        mode : str
            Either ``"3D"`` or ``"2D"``.
        """
        self.btn_3d.blockSignals(True)
        self.btn_2d.blockSignals(True)
        self.btn_3d.setChecked(mode == "3D")
        self.btn_2d.setChecked(mode == "2D")
        self.btn_3d.blockSignals(False)
        self.btn_2d.blockSignals(False)


class FibersWidget(QFrame):
    """Placeholder Fibers panel.

    Mirrors the layout of :class:`ClustersWidget` but with the count
    input, step arrows and Recovery button disabled — only the
    capture button is interactive today. Logic for the disabled
    controls will be wired up in follow-up work.
    """

    def __init__(self, *, parent=None):
        super().__init__(parent)
        self.setObjectName("fibersWidget")

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(8, 8, 8, 8)
        self.main_layout.setSpacing(8)

        self.title = QLabel("FIBERS")
        self.title.setObjectName("fibersTitle")
        self.main_layout.addWidget(self.title)

        std_h = 38

        self.btn_capture = QPushButton()
        self.btn_capture.setObjectName("fiberCaptureButton")
        self.btn_capture.setIcon(QIcon(str(ICONS_PATH / "capture.svg")))
        self.btn_capture.setIconSize(QSize(20, 14))
        self.btn_capture.setFixedSize(44, 44)
        self.btn_capture.setCursor(Qt.PointingHandCursor)
        self.btn_capture.setToolTip("Capture fibers")

        capture_row = QHBoxLayout()
        capture_row.setContentsMargins(0, 0, 0, 0)
        capture_row.setSpacing(0)
        capture_row.addWidget(self.btn_capture)
        capture_row.addStretch()
        self.main_layout.addLayout(capture_row)

        self.grid = QGridLayout()
        self.grid.setSpacing(6)
        self.grid.setContentsMargins(0, 0, 0, 0)

        self.count_input = QSpinBox()
        self.count_input.setObjectName("fiberCountInput")
        self.count_input.setRange(1, 100000)
        self.count_input.setValue(100)
        self.count_input.setButtonSymbols(QSpinBox.NoButtons)
        self.count_input.setFixedHeight(std_h)
        self.count_input.setMinimumWidth(56)
        self.count_input.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
        self.count_input.setEnabled(False)
        self.grid.addWidget(self.count_input, 0, 0)

        step_layout = QVBoxLayout()
        step_layout.setSpacing(2)
        self.btn_up = QPushButton("")
        self.btn_down = QPushButton("")
        self.btn_up.setIcon(QIcon(str(ICONS_PATH / "arrow_up.svg")))
        self.btn_down.setIcon(QIcon(str(ICONS_PATH / "arrow_down.svg")))
        self.btn_up.setIconSize(QSize(12, 12))
        self.btn_down.setIconSize(QSize(12, 12))
        self.btn_up.setObjectName("fiberStepButton")
        self.btn_down.setObjectName("fiberStepButton")
        self.btn_up.setFixedSize(33, (std_h // 2) - 1)
        self.btn_down.setFixedSize(33, (std_h // 2) - 1)
        self.btn_up.setEnabled(False)
        self.btn_down.setEnabled(False)
        step_layout.addWidget(self.btn_up)
        step_layout.addWidget(self.btn_down)
        self.grid.addLayout(step_layout, 0, 1)

        self.btn_recovery = QPushButton("Recovery")
        self.btn_recovery.setObjectName("fiberRecoveryButton")
        self.btn_recovery.setFixedHeight(std_h)
        self.btn_recovery.setMinimumWidth(70)
        self.btn_recovery.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn_recovery.setEnabled(False)
        self.grid.addWidget(self.btn_recovery, 0, 2)

        self.grid.setColumnStretch(0, 1)
        self.grid.setColumnStretch(1, 0)
        self.grid.setColumnStretch(2, 1)

        self.main_layout.addLayout(self.grid)


class ClustersWidget(QFrame):
    """Refined, compact version of the Cluster controls."""

    def __init__(self, *, parent=None):
        super().__init__(parent)
        self.setObjectName("clustersWidget")

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(8, 8, 8, 8)
        self.main_layout.setSpacing(8)

        self.title = QLabel("CLUSTERS")
        self.title.setObjectName("clustersTitle")
        self.main_layout.addWidget(self.title)

        self.grid = QGridLayout()
        self.grid.setSpacing(6)
        self.grid.setContentsMargins(0, 0, 0, 0)

        std_h = 38

        self.count_input = QSpinBox()
        self.count_input.setObjectName("clusterCountInput")
        self.count_input.setRange(1, 100000)
        self.count_input.setValue(100)
        self.count_input.setButtonSymbols(QSpinBox.NoButtons)
        self.count_input.setFixedHeight(std_h)
        self.count_input.setMinimumWidth(56)
        self.count_input.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
        self.count_input.returnPressed.connect(self._apply_clusters)
        self.grid.addWidget(self.count_input, 0, 0)

        step_layout = QVBoxLayout()
        step_layout.setSpacing(2)
        self.btn_up = QPushButton("")
        self.btn_down = QPushButton("")
        self.btn_up.setIcon(QIcon(str(ICONS_PATH / "arrow_up.svg")))
        self.btn_down.setIcon(QIcon(str(ICONS_PATH / "arrow_down.svg")))
        self.btn_up.setIconSize(QSize(12, 12))
        self.btn_down.setIconSize(QSize(12, 12))
        self.btn_up.setObjectName("clusterStepButton")
        self.btn_down.setObjectName("clusterStepButton")
        self.btn_up.setFixedSize(33, (std_h // 2) - 1)
        self.btn_down.setFixedSize(33, (std_h // 2) - 1)
        step_layout.addWidget(self.btn_up)
        step_layout.addWidget(self.btn_down)
        self.grid.addLayout(step_layout, 0, 1)

        self.btn_apply = QPushButton("Apply")
        self.btn_apply.setObjectName("clusterApplyButton")
        self.btn_apply.setFixedHeight(std_h)
        self.btn_apply.setMinimumWidth(50)
        self.btn_apply.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn_apply.clicked.connect(self._apply_clusters)
        self.grid.addWidget(self.btn_apply, 0, 2)

        self.btn_prev = QPushButton("Prev State")
        self.btn_next = QPushButton("Next State")
        self.btn_prev.setObjectName("clusterNavButton")
        self.btn_next.setObjectName("clusterNavButton")
        self.btn_prev.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
        self.btn_next.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
        self.btn_prev.setMinimumWidth(90)
        self.btn_next.setMinimumWidth(90)
        self.btn_prev.setFixedHeight(std_h)
        self.btn_next.setFixedHeight(std_h)
        self.btn_prev.clicked.connect(self._on_prev_state)
        self.btn_next.clicked.connect(self._on_next_state)

        self.grid.addWidget(self.btn_prev, 1, 0, 1, 2)
        self.grid.addWidget(self.btn_next, 1, 2)

        self.btn_settings = QToolButton()
        self.btn_settings.setObjectName("clusterSettingsButton")
        self.btn_settings.setFixedHeight(std_h)
        self.btn_settings.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
        self.btn_settings.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self.btn_settings.setPopupMode(QToolButton.InstantPopup)
        settings_icon_path = ICONS_PATH / "settings.svg"
        if settings_icon_path.exists():
            self.btn_settings.setIcon(QIcon(str(settings_icon_path)))
            self.btn_settings.setIconSize(QSize(24, 24))

        self.settings_menu = QMenu(self.btn_settings)
        self.settings_menu.setObjectName("clusterSettingsMenu")
        self.settings_menu.aboutToShow.connect(self._sync_settings_menu_width)

        for action_label in (
            "All",
            "None",
            "Swap",
            "Show",
            "Hide",
            "Delete",
            "Expand",
            "Collapse",
        ):
            action = QAction(action_label, self.btn_settings)

            action.triggered.connect(
                lambda checked=False, label=action_label: self._on_cluster_menu_action(
                    label
                )
            )

            self.settings_menu.addAction(action)

        self.btn_settings.setMenu(self.settings_menu)
        self.grid.addWidget(self.btn_settings, 2, 0, 1, 3)

        self.grid.setColumnStretch(0, 1)
        self.grid.setColumnStretch(1, 0)
        self.grid.setColumnStretch(2, 1)

        self.main_layout.addLayout(self.grid)

        self.btn_up.clicked.connect(self.count_input.stepUp)
        self.btn_down.clicked.connect(self.count_input.stepDown)

        for btn in [
            self.btn_apply,
            self.btn_prev,
            self.btn_next,
            self.btn_settings,
            self.btn_up,
            self.btn_down,
        ]:
            btn.setCursor(Qt.PointingHandCursor)

    def _sync_settings_menu_width(self):
        """Keep settings dropdown width aligned to its button width."""
        self.settings_menu.setMinimumWidth(self.btn_settings.width())

    def _apply_clusters(self, *, only_cluster=True):
        """Apply the clusters."""
        value = self.count_input.value()
        if 1 <= value <= self.count_input.maximum():
            self._remove_tractogram_visualizations()
            if only_cluster:
                state_manager.get_latest_state().tractogram_states = None
            visualization_manager.visualize_tractogram(nb_clusters=value)
            self._add_tractogram_visualizations()

    def _remove_tractogram_visualizations(self):
        """Remove the tractogram visualizations."""
        parent = self.parent()
        parent.parent().remove_visualization(
            visualization_manager.tractogram_visualizations,
            visualization_type="tractogram",
        )

    def _add_tractogram_visualizations(self):
        """Add the tractogram visualizations."""
        parent = self.parent()
        screen = parent.parent()
        screen.add_visualization(
            visualization_manager.tractogram_visualizations,
            visualization_type="tractogram",
        )
        screen._refresh_mesh_projection_if_active()

    def _on_prev_state(self):
        """Handle the 'Previous State' button click."""
        if state_manager.can_move_back():
            latest_state = state_manager.move_back()
            self.count_input.setMaximum(latest_state.max_clusters)
            self.count_input.setValue(latest_state.nb_clusters)
            self._apply_clusters(only_cluster=False)

    def _on_next_state(self):
        """Handle the 'Next State' button click."""
        if state_manager.can_move_next():
            latest_state = state_manager.move_next()
            self.count_input.setMaximum(latest_state.max_clusters)
            self.count_input.setValue(latest_state.nb_clusters)
            self._apply_clusters(only_cluster=False)

    def _on_cluster_menu_action(self, action_name):
        """Handle cluster settings menu actions."""
        if action_name == "All":
            visualization_manager.select_all_clusters()
        elif action_name == "None":
            visualization_manager.select_none_clusters()
        elif action_name == "Swap":
            visualization_manager.swap_clusters()
        elif action_name == "Show":
            visualization_manager.show_clusters()
        elif action_name == "Hide":
            visualization_manager.hide_clusters()
        elif action_name == "Delete":
            self._remove_tractogram_visualizations()
            visualization_manager.delete_clusters()
            self._add_tractogram_visualizations()
            return
        elif action_name == "Expand":
            self._remove_tractogram_visualizations()
            visualization_manager.expand_clusters()
            self._add_tractogram_visualizations()
            return
        elif action_name == "Collapse":
            self._remove_tractogram_visualizations()
            visualization_manager.collapse_clusters()
            self._add_tractogram_visualizations()
            return

        # Visibility-only mutations (Show/Hide/All/None/Swap) don't go through
        # remove/add of the tractogram visualization, so refresh the projection
        # explicitly to track the new visible-cluster set.
        self.parent().parent()._refresh_mesh_projection_if_active()


class RoiCreateWidget(QFrame):
    """ROI EDIT panel shown while drawing in 2D mode.

    Layout matches the design mock: a title, a four-button toolbar
    (square / circle / brush / eraser), and a read-only Properties
    panel listing the active ROI's name, visibility, type, voxel
    position and color swatch.

    Only the circle tool is wired up today (sphere ROI); the other
    three are placeholders disabled at the UI level so the design
    fits while the rasterizers for those modes are added later.

    A single ROI is drawn per session — re-dragging overwrites the
    same draft. Switching to 3D commits and runs the streamline
    filter; there is no explicit Save button.
    """

    shape_changed = Signal(str)
    edit_requested = Signal(str)

    def __init__(self, *, parent=None):
        super().__init__(parent)
        self.setObjectName("roiCreateWidget")

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(12, 12, 12, 12)
        self.main_layout.setSpacing(12)

        self.title = QLabel("ROI EDIT")
        self.title.setObjectName("roiCreateTitle")
        self.main_layout.addWidget(self.title)

        self._build_toolbar()
        self._build_properties()
        self._build_existing_list()

    def _build_toolbar(self):
        """Toolbar of 4 icon buttons.

        ``square_roi`` and ``sphere_roi`` are wired to the rectangle
        and sphere rasterizers respectively. ``edit_roi`` (brush) and
        ``erase_roi`` are placeholders for future tools and stay
        disabled at the UI level.
        """
        toolbar_row = QHBoxLayout()
        toolbar_row.setSpacing(8)

        self._tool_group = QButtonGroup(self)
        self._tool_group.setExclusive(True)

        tools = [
            ("rectangle", "square_roi.svg", "rectangle", "Rectangle", True),
            ("circle", "sphere_roi.svg", "sphere", "Sphere", True),
            ("brush", "edit_roi.svg", None, "Brush (coming soon)", False),
            ("eraser", "erase_roi.svg", None, "Eraser (coming soon)", False),
        ]
        self._tool_buttons = {}
        self._tool_shape = {}
        for key, icon_file, shape, tooltip, enabled in tools:
            btn = QPushButton()
            btn.setObjectName(f"roiTool_{key}")
            btn.setProperty("class", "roiToolButton")
            btn.setIcon(QIcon(str(ICONS_PATH / icon_file)))
            btn.setIconSize(QSize(20, 20))
            btn.setCheckable(True)
            btn.setEnabled(enabled)
            btn.setFixedSize(44, 44)
            btn.setCursor(Qt.PointingHandCursor if enabled else Qt.ForbiddenCursor)
            btn.setToolTip(tooltip)
            if key == "circle":
                btn.setChecked(True)
            self._tool_group.addButton(btn)
            self._tool_buttons[key] = btn
            self._tool_shape[key] = shape
            toolbar_row.addWidget(btn)
        toolbar_row.addStretch()
        self.main_layout.addLayout(toolbar_row)

        # Each enabled tool emits its shape name on selection; the
        # rasterizer in _on_roi_drawn branches on that name.
        for key, btn in self._tool_buttons.items():
            shape = self._tool_shape[key]
            if shape is None:
                continue
            btn.toggled.connect(
                lambda checked, s=shape: checked and self.shape_changed.emit(s)
            )

    def _build_properties(self):
        """Properties pane: read-only labels showing the active ROI's metadata."""
        props_title = QLabel("Properties")
        props_title.setObjectName("roiPropsTitle")
        self.main_layout.addWidget(props_title)

        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(6)
        grid.setColumnStretch(1, 1)

        def _value_label(text):
            label = QLabel(text)
            label.setObjectName("roiPropValue")
            return label

        def _key_label(text):
            label = QLabel(text)
            label.setObjectName("roiPropKey")
            return label

        self._name_value = _value_label("–")
        self._visibility_value = _value_label("On")
        self._type_value = _value_label("–")
        self._position_value = _value_label("–")

        # Color swatch: a thin filled bar driven by stylesheet
        # background-color. Default placeholder color until a draw
        # commits a real one.
        self._color_swatch = QFrame()
        self._color_swatch.setObjectName("roiPropColor")
        self._color_swatch.setFixedHeight(14)
        self._color_swatch.setStyleSheet(
            "background-color: rgb(80, 80, 80); border-radius: 7px;"
        )

        rows = [
            ("Name", self._name_value),
            ("Visibility", self._visibility_value),
            ("Type", self._type_value),
            ("Position", self._position_value),
            ("Color", self._color_swatch),
        ]
        for row_index, (key, value_widget) in enumerate(rows):
            grid.addWidget(_key_label(key), row_index, 0)
            grid.addWidget(value_widget, row_index, 1)
        self.main_layout.addLayout(grid)

    def _build_existing_list(self):
        """List of ROIs already drawn (across the whole session).

        Each row carries the ROI's color as a small disk icon and
        its name as the label. Selecting a row makes that ROI the
        active edit target; the parent screen sets it as the draft
        so the next drag rewrites that ROI in place.
        """
        list_title = QLabel("Existing ROIs")
        list_title.setObjectName("roiPropsTitle")
        self.main_layout.addWidget(list_title)

        self._existing_list = QListWidget()
        self._existing_list.setObjectName("roiExistingList")
        self._existing_list.setSelectionMode(QListWidget.SingleSelection)
        self._existing_list.setUniformItemSizes(True)
        self._existing_list.setIconSize(QSize(14, 14))
        self._existing_list.setMaximumHeight(140)
        self._existing_list.itemSelectionChanged.connect(
            self._on_existing_selection_changed
        )
        self.main_layout.addWidget(self._existing_list)

        self._existing_empty_label = QLabel("No ROIs yet")
        self._existing_empty_label.setObjectName("roiPropValue")
        self._existing_empty_label.setVisible(True)
        self.main_layout.addWidget(self._existing_empty_label)

    @staticmethod
    def _color_disk_icon(color):
        """Return a small filled-circle QIcon tinted with ``color``.

        ``color`` is an RGB triple in [0, 1]. Used to mirror the swatch
        shown in the Properties pane on each list row.
        """
        pixmap = QPixmap(14, 14)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        try:
            r, g, b = (int(max(0.0, min(1.0, float(c))) * 255) for c in color[:3])
            painter.setBrush(QColor(r, g, b))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(1, 1, 12, 12)
        finally:
            painter.end()
        return QIcon(pixmap)

    def refresh_existing_rois(self, items):
        """Rebuild the existing-ROIs list.

        Parameters
        ----------
        items : list of dict
            Each dict carries ``name`` (str) and ``color`` (RGB
            triple in [0, 1]) for one ROI. Pass an empty list to
            show the placeholder. Selection state is preserved
            across calls when the previously-selected name still
            exists in the new list.
        """
        previous_selected = self.selected_roi_name()

        self._existing_list.blockSignals(True)
        try:
            self._existing_list.clear()
            for entry in items:
                name = str(entry.get("name", "ROI"))
                # Use an explicit None check rather than ``or``: when
                # the visualization manager returns a numpy array,
                # the ``or`` truthiness test raises
                # "truth value of an array is ambiguous".
                color = entry.get("color")
                if color is None:
                    color = (0.5, 0.5, 0.5)
                item = QListWidgetItem(self._color_disk_icon(color), name)
                item.setData(Qt.UserRole, name)
                self._existing_list.addItem(item)

            if previous_selected is not None:
                for row in range(self._existing_list.count()):
                    item = self._existing_list.item(row)
                    if item.data(Qt.UserRole) == previous_selected:
                        item.setSelected(True)
                        self._existing_list.setCurrentItem(item)
                        break
        finally:
            self._existing_list.blockSignals(False)

        self._existing_empty_label.setVisible(self._existing_list.count() == 0)

    def selected_roi_name(self):
        """Return the name of the currently-selected ROI, or None."""
        item = self._existing_list.currentItem()
        if item is None or not item.isSelected():
            return None
        return item.data(Qt.UserRole)

    def clear_existing_selection(self):
        """Deselect any selected row without firing edit_requested."""
        self._existing_list.blockSignals(True)
        try:
            self._existing_list.clearSelection()
            self._existing_list.setCurrentItem(None)
        finally:
            self._existing_list.blockSignals(False)

    def _on_existing_selection_changed(self):
        """Forward the new selection — or its absence — to the parent."""
        name = self.selected_roi_name()
        # Empty string signals "no selection — next drag is a new ROI";
        # a non-empty name asks the parent to make that ROI the draft.
        self.edit_requested.emit(name or "")

    def current_shape(self):
        """Return the active drawing shape name.

        Reads from whichever toolbar button is currently checked.
        Square = rectangle, Circle = sphere; brush and eraser have
        no rasterizer yet and fall through to the default sphere.
        """
        for key, btn in self._tool_buttons.items():
            if btn.isChecked():
                shape = self._tool_shape.get(key)
                if shape is not None:
                    return shape
        return "sphere"

    def set_properties(
        self, *, name=None, visibility=None, type_=None, position=None, color=None
    ):
        """Update the properties pane with the active ROI's metadata.

        All arguments are optional; pass only the fields that
        actually changed. ``position`` is expected to be an iterable
        of three voxel indices; ``color`` is an RGB triple in [0, 1].
        """
        if name is not None:
            self._name_value.setText(str(name))
        if visibility is not None:
            self._visibility_value.setText("On" if visibility else "Off")
        if type_ is not None:
            self._type_value.setText(str(type_))
        if position is not None:
            try:
                px, py, pz = (int(round(v)) for v in position)
                self._position_value.setText(f"[{px} {py} {pz}]")
            except Exception:
                self._position_value.setText(str(position))
        if color is not None:
            r, g, b = (int(max(0, min(1, c)) * 255) for c in color[:3])
            self._color_swatch.setStyleSheet(
                f"background-color: rgb({r}, {g}, {b}); border-radius: 7px;"
            )

    def reset_properties(self):
        """Clear the properties pane back to placeholder values."""
        self._name_value.setText("–")
        self._visibility_value.setText("On")
        self._type_value.setText("–")
        self._position_value.setText("–")
        self._color_swatch.setStyleSheet(
            "background-color: rgb(80, 80, 80); border-radius: 7px;"
        )


class LeftSectionWidget(QFrame):
    """The Sidebar container that holds the control modules."""

    def __init__(self, *, parent=None):
        super().__init__(parent)
        self.setObjectName("interactionLeftSection")

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(8, 8, 8, 8)
        self.main_layout.setSpacing(10)

        self.view_mode_widget = ViewModeWidget(parent=self)
        self.main_layout.addWidget(self.view_mode_widget)

        self.fibers_box = FibersWidget(parent=self)
        self.main_layout.addWidget(self.fibers_box)

        self.clusters_box = ClustersWidget(parent=self)
        self.main_layout.addWidget(self.clusters_box)

        self.roi_input_widget = RoiInputWidget(parent=self)
        self.main_layout.addWidget(self.roi_input_widget)

        self.roi_create_widget = RoiCreateWidget(parent=self)
        self.roi_create_widget.setVisible(False)
        self.main_layout.addWidget(self.roi_create_widget)

        self.main_layout.addStretch()

        self._track_isolation_active = False

    def set_track_isolation_active(self, active):
        """Hide cluster/ROI panels while a captured track is isolated."""
        self._track_isolation_active = bool(active)
        self.update_controls_for_visualization()

    def update_controls_for_visualization(self):
        """Show/hide controls depending on visualization type and view mode."""
        is_3d = state_manager.view_mode == "3D"
        is_create_mode = state_manager.roi_create_mode is not None
        has_tractogram_input = input_manager.has_tractogram
        isolating = self._track_isolation_active
        self.fibers_box.setVisible(is_3d and has_tractogram_input and not isolating)
        self.clusters_box.setVisible(is_3d and has_tractogram_input and not isolating)
        self.roi_input_widget.setVisible(is_3d and not is_create_mode and not isolating)
        self.roi_create_widget.setVisible(is_create_mode and not isolating)

        if has_tractogram_input:
            self._sync_clusters_from_latest_state()

    def _sync_clusters_from_latest_state(self):
        """Populate cluster widget fields from your state manager."""
        if state_manager.has_states():
            state = state_manager.get_latest_state()
            self.clusters_box.count_input.setValue(int(state.nb_clusters))
