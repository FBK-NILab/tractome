import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

UP_ARROW = os.path.join(os.path.dirname(__file__), "assets", "up_arrow.svg").replace(
    "\\", "/"
)
DOWN_ARROW = os.path.join(
    os.path.dirname(__file__), "assets", "down_arrow.svg"
).replace("\\", "/")

STYLE_SHEET = (
    """
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
    color: #333333;
}
QPushButton {
    background-color: #E1E1E1;
    border: 1px solid #ADADAD;
    padding: 5px;
    border-radius: 3px;
    min-height: 20px;
    color: #333333;
}
QPushButton:hover, QSpinBox::up-button:hover, QSpinBox::down-button:hover {
    background-color: #D1D1D1;
    border: 1px solid #999999;
}
QPushButton:pressed {
    background-color: #C1C1C1;
}
QSpinBox {
    min-height: 20px;
    border: 1px solid #ADADAD;
    border-radius: 3px;
    color: #333333;
    padding: 5px;
    background-color: #FFFFFF;
}
QSpinBox::up-button {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    border: 1px solid #ADADAD;
    width: 10px;
    height: 10px;
    padding: 2px;
    background-color: #E1E1E1;
    border-top-right-radius: 2px;
}

QSpinBox::down-button {
    subcontrol-origin: padding;
    subcontrol-position: bottom right;
    border: 1px solid #ADADAD;
    width: 10px;
    height: 10px;
    padding: 2px;
    background-color: #E1E1E1;
    border-bottom-right-radius: 2px;
}
QTableWidget {
    color: #333333;
    background-color: #FFFFFF;
}
QRadioButton {
    color: #333333;
}
QLabel {
    color: #333333;
}
#TitleLabel {
    font-size: 24px;
    font-weight: bold;
    padding-bottom: 10px;
}
#SliderValueLabel {
    font-weight: bold;
    color: #333333;
    text-align: right;
    margin-left: 5px;
}
#SliderMinMaxLabel {
    font-size: 10px;
    color: #666666;
    min-width: 30px;
    max-width: 40px;
}
QSlider::groove:horizontal {
    border: 1px solid #CCCCCC;
    height: 6px;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #F0F0F0, stop:1 #E0E0E0);
    margin: 0px 0;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #FFFFFF, stop:1 #D0D0D0);
    border: 1px solid #ADADAD;
    width: 16px;
    margin: -3px 0;
    border-radius: 8px;
}
QSlider {
    height: 20px;
}
QSlider::handle:horizontal:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #F5F5F5, stop:1 #C5C5C5);
    border: 1px solid #999999;
}
QSlider::handle:horizontal:pressed {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #E5E5E5, stop:1 #B5B5B5);
}
"""
    + f"""
QSpinBox::up-arrow {{
    image: url('{str(UP_ARROW)}');
    width: 20px;
    height: 20px;
}}
QSpinBox::down-arrow {{
    image: url('{str(DOWN_ARROW)}');
    width: 20px;
    height: 20px;
}}
"""
)


def _create_slider(
    slider_label,
    min_val,
    max_val,
    object_suffix,
    control_type="checkbox",
    default_value=None,
):
    """Create a slider with its layout.

    Parameters
    ----------
    slider_label : str
        The label text for the slider (e.g., "X Slice:", "Clusters:").
    min_val : int
        The minimum value for the slider.
    max_val : int
        The maximum value for the slider.
    object_suffix : str
        The suffix for the slider object name.
    control_type : str, optional
        Type of control widget: "checkbox", "radio", or "none". Default is "checkbox".
    default_value : int, optional
        The default value for the slider. If None, uses the middle value
        between min and max.

    Returns
    -------
    tuple
        A tuple containing
        (main_layout, slider, value_label, control_widget, max_label).
    """
    # Calculate default value
    if default_value is None:
        default_value = (max_val + min_val) // 2

    # Create labels
    label = QLabel(slider_label)
    label.setObjectName("SliderLabel")
    value_label = QLabel(str(default_value))
    value_label.setObjectName("SliderValueLabel")
    value_label.setMinimumWidth(40)

    # Create control widget based on type
    control_widget = None
    if control_type.lower() == "radio":
        control_widget = QRadioButton()
        control_widget.setChecked(True)  # Default to checked
    elif control_type.lower() == "checkbox":
        control_widget = QCheckBox()
        control_widget.setChecked(True)  # Default to checked
    # If control_type is "none" or anything else, no control widget is created

    # Create slider
    slider = QSlider(Qt.Horizontal)
    slider.setMinimum(min_val)
    slider.setMaximum(max_val)
    slider.setValue(default_value)
    slider.setObjectName(f"{object_suffix}Slider")
    slider.setMaximumHeight(20)

    # Create vertical layout for two-line slider controls
    main_layout = QVBoxLayout()
    main_layout.setSpacing(2)
    main_layout.setContentsMargins(0, 0, 0, 0)

    # First line: Label, current value, and control widget (if any)
    label_value_layout = QHBoxLayout()
    label_value_layout.setSpacing(5)
    label_value_layout.addWidget(label)
    label_value_layout.addWidget(value_label)
    label_value_layout.addStretch()
    if control_widget is not None:
        label_value_layout.addWidget(control_widget)

    # Second line: Min, slider, max
    slider_layout = QHBoxLayout()
    slider_layout.setSpacing(5)
    min_label = QLabel(str(min_val))
    min_label.setObjectName("SliderMinMaxLabel")
    max_label = QLabel(str(max_val))
    max_label.setObjectName("SliderMinMaxLabel")

    slider_layout.addWidget(min_label)
    slider_layout.addWidget(slider, 1)
    slider_layout.addWidget(max_label)
    slider_layout.addStretch()

    # Add both lines to main layout
    main_layout.addLayout(label_value_layout)
    main_layout.addLayout(slider_layout)

    # Connect slider to update value label
    slider.valueChanged.connect(lambda val: value_label.setText(str(val)))

    return main_layout, slider, value_label, control_widget, max_label


def create_slice_sliders(
    min_vals=(0, 0, 0),
    max_vals=(100, 100, 100),
    control_type="checkbox",
    default_vals=None,
):
    """Create slice control sliders for x, y, z axes.

    Parameters
    ----------
    min_vals : tuple, list, or ndarray, optional
        The minimum values for x, y, z sliders respectively. Default is (0, 0, 0).
    max_vals : tuple, list, or ndarray, optional
        The maximum values for x, y, z sliders respectively. Default is (100, 100, 100).
    control_type : str, optional
        Type of control widget: "checkbox" or "radio". Default is "checkbox".
    default_vals : tuple, list, ndarray, or None, optional
        The default values for x, y, z sliders respectively. If None, uses
        the middle value between min and max for each slider.

    Returns
    -------
    tuple
        A tuple containing (widget, sliders_tuple, controls_tuple, button_group).
        Note: button_group is None when using checkboxes.
    """
    slice_widget = QGroupBox("Image Controls")
    slice_layout = QVBoxLayout()
    slice_layout.setContentsMargins(10, 20, 5, 5)
    slice_layout.setSpacing(10)
    slice_widget.setLayout(slice_layout)

    # Create sliders for X, Y, Z axes
    axes = ["X", "Y", "Z"]
    sliders = []
    controls = []
    button_group = None

    # Create button group for radio buttons to ensure only one can be selected
    if control_type.lower() == "radio":
        button_group = QButtonGroup()

    for i, axis in enumerate(axes):
        # Get default value for this axis (if provided)
        default_value = None if default_vals is None else default_vals[i]

        layout, slider, value_label, control_widget, _ = _create_slider(
            f"{axis} Slice:",
            min_vals[i],
            max_vals[i],
            axis,
            control_type,
            default_value,
        )
        slice_layout.addLayout(layout)
        sliders.append(slider)
        controls.append(control_widget)

        # Add radio buttons to button group
        if button_group is not None:
            button_group.addButton(control_widget)

    return (
        slice_widget,
        (sliders[0], sliders[1], sliders[2]),
        (controls[0], controls[1], controls[2]),
        button_group,
    )


def create_clusters_slider(default_value=250):
    """Create a clusters input control.

    Parameters
    ----------
    default_value : int, optional
        The default value for the input.

    Returns
    -------
    tuple
        A tuple containing (widget, input_widget, apply_button, prev_button,
        next_button, history_table).
    """
    tractogram_widget = QGroupBox("Tractogram Controls")
    tractogram_layout = QVBoxLayout()
    tractogram_layout.setContentsMargins(10, 20, 5, 5)
    tractogram_layout.setSpacing(10)
    tractogram_widget.setLayout(tractogram_layout)

    # Create label
    label = QLabel("Clusters:")
    label.setObjectName("SliderLabel")
    tractogram_layout.addWidget(label)

    # Create input and button layout
    input_layout = QHBoxLayout()
    cluster_input = QSpinBox()
    cluster_input.setMinimum(1)
    cluster_input.setMaximum(1000)
    cluster_input.setValue(default_value)
    apply_button = QPushButton("Apply")
    input_layout.addWidget(cluster_input)
    input_layout.addWidget(apply_button)
    tractogram_layout.addLayout(input_layout)

    # Create state management buttons
    button_layout = QHBoxLayout()
    prev_button = QPushButton("Prev. State")
    next_button = QPushButton("Next State")
    button_layout.addWidget(prev_button)
    button_layout.addWidget(next_button)
    tractogram_layout.addLayout(button_layout)

    # Create history table
    history_table = QTableWidget()
    history_table.setRowCount(10)
    history_table.setColumnCount(2)
    history_table.setHorizontalHeaderLabels(["# of Clusters", "# of Fibers"])
    history_table.verticalHeader().setVisible(False)
    history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    tractogram_layout.addWidget(history_table)

    return (
        tractogram_widget,
        cluster_input,
        apply_button,
        prev_button,
        next_button,
        history_table,
    )


def create_roi_controls(rois):
    """Create ROI visibility controls.

    Parameters
    ----------
    rois : Sequence[str]
        Paths or names of ROIs to list.

    Returns
    -------
    tuple
        A tuple containing (widget, roi_checkboxes).
    """
    roi_widget = QGroupBox("ROI Controls")
    roi_layout = QVBoxLayout()
    roi_layout.setContentsMargins(10, 20, 5, 5)
    roi_layout.setSpacing(10)
    roi_widget.setLayout(roi_layout)

    checkboxes = []
    for roi in rois:
        row_layout = QHBoxLayout()
        row_layout.setSpacing(5)
        label = QLabel(os.path.basename(roi))
        label.setObjectName("SliderLabel")
        checkbox = QCheckBox()
        checkbox.setChecked(True)
        row_layout.addWidget(label)
        row_layout.addStretch()
        row_layout.addWidget(checkbox)
        roi_layout.addLayout(row_layout)
        checkboxes.append(checkbox)

    return roi_widget, checkboxes


def create_cluster_selection_buttons():
    """Create the cluster selection buttons.

    Returns
    -------
    tuple
        A tuple containing (widget, all_button, none_button, swap_button,
        delete_button).
    """
    cluster_selection_widget = QGroupBox("Cluster Selection")
    cluster_selection_layout = QVBoxLayout()
    cluster_selection_layout.setContentsMargins(10, 20, 10, 5)
    cluster_selection_layout.setSpacing(10)
    cluster_selection_widget.setLayout(cluster_selection_layout)

    all_button = QPushButton("All")
    none_button = QPushButton("None")
    swap_button = QPushButton("Swap")
    delete_button = QPushButton("Del")

    row1_layout = QHBoxLayout()
    row1_layout.setSpacing(10)
    row1_layout.addWidget(all_button)
    row1_layout.addWidget(none_button)
    row1_layout.addWidget(swap_button)
    row1_layout.addWidget(delete_button)
    cluster_selection_layout.addLayout(row1_layout)

    row2_layout = QHBoxLayout()
    row2_layout.setSpacing(10)
    expand_button = QPushButton("Exp")
    collapse_button = QPushButton("Coll")
    show_button = QPushButton("Show")
    hide_button = QPushButton("Hide")
    row2_layout.addWidget(expand_button)
    row2_layout.addWidget(collapse_button)
    row2_layout.addWidget(show_button)
    row2_layout.addWidget(hide_button)
    cluster_selection_layout.addLayout(row2_layout)

    return (
        cluster_selection_widget,
        all_button,
        none_button,
        swap_button,
        delete_button,
        expand_button,
        collapse_button,
        show_button,
        hide_button,
    )


def update_history_table(table, data, current_index=None):
    """Update the history table with the latest data.

    Parameters
    ----------
    table : QTableWidget
        The table to update.
    data : list of ClusterState
        A list of ClusterState objects.
    current_index : int, optional
        The index of the currently selected state to highlight.
    """
    table.clearContents()
    table.setRowCount(len(data))
    for i, state in enumerate(data):
        item0 = QTableWidgetItem(str(state.nb_clusters))
        item1 = QTableWidgetItem(str(len(state.streamline_ids)))
        item0.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        item1.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        if current_index is not None and i == current_index:
            highlight_color = QColor(173, 216, 230)  # Light blue
            item0.setBackground(highlight_color)
            item1.setBackground(highlight_color)
        table.setItem(i, 0, item0)
        table.setItem(i, 1, item1)


def _create_left_panel(fix_width=250, title="Tractome 2.0"):
    """Create the left panel widget.

    Parameters
    ----------
    fix_width : int, optional
        The fixed width of the panel.
    title : str, optional
        The title of the panel.

    Returns
    -------
    QWidget
        The left panel widget.
    """
    left_panel_widget = QWidget()
    left_layout = QVBoxLayout(left_panel_widget)
    left_layout.setContentsMargins(10, 10, 10, 10)
    left_layout.setSpacing(10)

    left_panel_widget.setFixedWidth(fix_width)

    title_label = QLabel(title)
    title_label.setObjectName("TitleLabel")
    left_layout.addWidget(title_label)

    left_layout.addStretch()

    return left_panel_widget


def _create_right_panel(viz_window):
    """Create the right panel widget.

    Parameters
    ----------
    viz_window : QWidget
        The main visualization window.

    Returns
    -------
    tuple
        A tuple containing (right_panel_widget, button_3d, button_2d, reset_button).
    """
    right_panel_widget = QWidget()
    right_layout = QVBoxLayout(right_panel_widget)
    right_layout.setContentsMargins(0, 10, 10, 10)
    right_layout.setSpacing(10)

    toggle_button_layout = QHBoxLayout()
    toggle_button_layout.addStretch(1)

    toggle_suggestion_button = QPushButton("Toggle Suggestion")
    reset_button = QPushButton("Reset View")
    separator = QLabel("|")
    button_3d = QPushButton("3D VIEW")
    button_2d = QPushButton("2D VIEW")

    toggle_button_layout.addWidget(toggle_suggestion_button)
    toggle_button_layout.addWidget(reset_button)
    toggle_button_layout.addWidget(separator)
    toggle_button_layout.addWidget(button_3d)
    toggle_button_layout.addWidget(button_2d)
    right_layout.addLayout(toggle_button_layout)

    right_layout.addWidget(viz_window)

    return (
        right_panel_widget,
        button_3d,
        button_2d,
        reset_button,
        toggle_suggestion_button,
    )


def create_ui(viz_window):
    """Create the main UI layout with left and right panels.

    Parameters
    ----------
    viz_window : QWidget
        The main visualization window.

    Returns
    -------
    tuple
        A tuple containing
        (main_widget, left_panel, right_panel, button_3d, button_2d, reset_button).
    """
    main_widget = QWidget()
    main_layout = QHBoxLayout(main_widget)

    left_panel = _create_left_panel()
    right_panel, button_3d, button_2d, reset_button, toggle_suggestion_button = (
        _create_right_panel(viz_window)
    )

    main_layout.addWidget(left_panel)
    main_layout.addWidget(right_panel, 1)

    return (
        main_widget,
        left_panel,
        right_panel,
        button_3d,
        button_2d,
        reset_button,
        toggle_suggestion_button,
    )


def create_mesh_controls():
    """Create mesh control widgets for opacity, visibility, and mesh mode selection.

    Returns
    -------
    tuple
        A tuple containing (widget, opacity_slider, visibility_checkbox,
        radio_group, photogram_radio, normals_radio).
    """
    mesh_widget = QGroupBox("Mesh Controls")
    mesh_layout = QVBoxLayout()
    mesh_layout.setContentsMargins(10, 20, 5, 5)
    mesh_layout.setSpacing(10)
    mesh_widget.setLayout(mesh_layout)

    # Slider for opacity with visibility checkbox on the right
    (
        opacity_layout,
        opacity_slider,
        opacity_value_label,
        visibility_checkbox,
        _,
    ) = _create_slider(
        "Opacity:",
        0,
        100,
        "Opacity",
        "checkbox",
        100,
    )
    mesh_layout.addLayout(opacity_layout)

    radio_layout = QHBoxLayout()
    radio_layout.setSpacing(10)
    normals_radio = QRadioButton("Normals")
    normals_radio.setChecked(True)
    photographic_radio = QRadioButton("Photographic")
    radio_group = QButtonGroup(mesh_widget)
    radio_group.addButton(normals_radio)
    radio_group.addButton(photographic_radio)
    radio_layout.addWidget(normals_radio)
    radio_layout.addWidget(photographic_radio)
    mesh_layout.addLayout(radio_layout)

    return (
        mesh_widget,
        opacity_slider,
        visibility_checkbox,
        radio_group,
        normals_radio,
        photographic_radio,
    )
