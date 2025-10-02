from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

STYLE_SHEET = """
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
        A tuple containing (main_layout, slider, value_label, control_widget).
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
    """Create a clusters slider.

    Parameters
    ----------
    default_value : int, optional
        The default value for the slider. Default is 250.

    Returns
    -------
    tuple
        A tuple containing (widget, slider, value_label, prev_button,
        save_button, history_table, max_label).
    """
    tractogram_widget = QGroupBox("Tractogram Controls")
    tractogram_layout = QVBoxLayout()
    tractogram_layout.setContentsMargins(10, 20, 5, 5)
    tractogram_layout.setSpacing(10)
    tractogram_widget.setLayout(tractogram_layout)

    layout, slider, value_label, _, max_label = _create_slider(
        "Clusters:", 1, 1000, "Clusters", "none", default_value
    )

    tractogram_layout.addLayout(layout)

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
    history_table.setHorizontalHeaderLabels(["# clusters", "# streamlines"])
    tractogram_layout.addWidget(history_table)

    return (
        tractogram_widget,
        slider,
        value_label,
        prev_button,
        next_button,
        history_table,
        max_label,
    )


def create_cluster_selection_buttons():
    """Create the cluster selection buttons.

    Returns
    -------
    tuple
        A tuple containing (widget, all_button, none_button, swap_button,
        delete_button).
    """
    cluster_selection_widget = QGroupBox("Cluster Selection")
    cluster_selection_layout = QHBoxLayout()
    cluster_selection_layout.setContentsMargins(10, 20, 10, 5)
    cluster_selection_layout.setSpacing(10)
    cluster_selection_widget.setLayout(cluster_selection_layout)

    all_button = QPushButton("All")
    none_button = QPushButton("None")
    swap_button = QPushButton("Swap")
    delete_button = QPushButton("Del")

    cluster_selection_layout.addWidget(all_button)
    cluster_selection_layout.addWidget(none_button)
    cluster_selection_layout.addWidget(swap_button)
    cluster_selection_layout.addWidget(delete_button)

    return (
        cluster_selection_widget,
        all_button,
        none_button,
        swap_button,
        delete_button,
    )


def update_cluster_slider(slider, max_label, max_value):
    """Update the cluster slider's max value and text.

    Parameters
    ----------
    slider : QSlider
        The slider to update.
    max_label : QLabel
        The label showing the max value of the slider.
    max_value : int
        The new maximum value for the slider.
    """
    slider.setMaximum(max_value)
    max_label.setText(str(max_value))


def update_history_table(table, data):
    """Update the history table with the latest data.

    Parameters
    ----------
    table : QTableWidget
        The table to update.
    data : list of ClusterState
        A list of ClusterState objects.
    """
    table.clearContents()
    table.setRowCount(len(data))
    for i, state in enumerate(data):
        table.setItem(i, 0, QTableWidgetItem(str(state.nb_clusters)))
        table.setItem(i, 1, QTableWidgetItem(str(len(state.streamline_ids))))


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
        A tuple containing (right_panel_widget, button_3d, button_2d).
    """
    right_panel_widget = QWidget()
    right_layout = QVBoxLayout(right_panel_widget)
    right_layout.setContentsMargins(0, 10, 10, 10)
    right_layout.setSpacing(10)

    toggle_button_layout = QHBoxLayout()
    toggle_button_layout.addStretch(1)

    button_3d = QPushButton("3D VIEW")
    button_2d = QPushButton("2D VIEW")

    toggle_button_layout.addWidget(button_3d)
    toggle_button_layout.addWidget(button_2d)
    right_layout.addLayout(toggle_button_layout)

    right_layout.addWidget(viz_window)

    return right_panel_widget, button_3d, button_2d


def create_ui(viz_window):
    """Create the main UI layout with left and right panels.

    Parameters
    ----------
    viz_window : QWidget
        The main visualization window.

    Returns
    -------
    tuple
        A tuple containing (main_widget, left_panel, right_panel, button_3d, button_2d).
    """
    main_widget = QWidget()
    main_layout = QHBoxLayout(main_widget)

    left_panel = _create_left_panel()
    right_panel, button_3d, button_2d = _create_right_panel(viz_window)

    main_layout.addWidget(left_panel)
    main_layout.addWidget(right_panel, 1)

    return main_widget, left_panel, right_panel, button_3d, button_2d
