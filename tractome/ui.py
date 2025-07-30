from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
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
#SliceValueLabel {
    font-weight: bold;
    color: #333333;
    text-align: right;
    margin-left: 5px;
}
#SliceMinMaxLabel {
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


def _create_slider(axis_name, min_val, max_val, object_suffix):
    """Create a slice slider with its layout.

    Parameters
    ----------
    axis_name : str
        The name of the axis (e.g., "X", "Y", "Z").
    min_val : int
        The minimum value for the slider.
    max_val : int
        The maximum value for the slider.
    object_suffix : str
        The suffix for the slider object name.

    Returns
    -------
    tuple
        A tuple containing (main_layout, slider, value_label, checkbox).
    """
    # Create labels
    label = QLabel(f"{axis_name} Slice:")
    label.setObjectName("SliceLabel")
    value_label = QLabel(str((max_val + min_val) // 2))
    value_label.setObjectName("SliceValueLabel")
    value_label.setMinimumWidth(40)

    # Create checkbox
    checkbox = QCheckBox()
    checkbox.setChecked(True)  # Default to checked

    # Create slider
    slider = QSlider(Qt.Horizontal)
    slider.setMinimum(min_val)
    slider.setMaximum(max_val)
    slider.setValue((max_val + min_val) // 2)
    slider.setObjectName(f"{object_suffix}SliceSlider")
    slider.setMaximumHeight(20)

    # Create vertical layout for two-line slider controls
    main_layout = QVBoxLayout()
    main_layout.setSpacing(2)
    main_layout.setContentsMargins(0, 0, 0, 0)

    # First line: Label, current value, and checkbox
    label_value_layout = QHBoxLayout()
    label_value_layout.setSpacing(5)
    label_value_layout.addWidget(label)
    label_value_layout.addWidget(value_label)
    label_value_layout.addStretch()
    label_value_layout.addWidget(checkbox)

    # Second line: Min, slider, max
    slider_layout = QHBoxLayout()
    slider_layout.setSpacing(5)
    min_label = QLabel(str(min_val))
    min_label.setObjectName("SliceMinMaxLabel")
    max_label = QLabel(str(max_val))
    max_label.setObjectName("SliceMinMaxLabel")

    slider_layout.addWidget(min_label)
    slider_layout.addWidget(slider, 1)
    slider_layout.addWidget(max_label)
    slider_layout.addStretch()

    # Add both lines to main layout
    main_layout.addLayout(label_value_layout)
    main_layout.addLayout(slider_layout)

    # Connect slider to update value label
    slider.valueChanged.connect(lambda val: value_label.setText(str(val)))

    return main_layout, slider, value_label, checkbox


def create_slice_sliders(min_vals=(0, 0, 0), max_vals=(100, 100, 100)):
    """Create slice control sliders for x, y, z axes.

    Parameters
    ----------
    min_vals : tuple, list, or ndarray, optional
        The minimum values for x, y, z sliders respectively. Default is (0, 0, 0).
    max_vals : tuple, list, or ndarray, optional
        The maximum values for x, y, z sliders respectively. Default is (100, 100, 100).

    Returns
    -------
    tuple
        A tuple containing (widget, sliders_tuple, checkboxes_tuple).
    """
    slice_widget = QGroupBox("Image Controls")
    slice_layout = QVBoxLayout()
    slice_layout.setContentsMargins(10, 20, 5, 5)
    slice_layout.setSpacing(10)
    slice_widget.setLayout(slice_layout)

    # Create sliders for X, Y, Z axes
    axes = ["X", "Y", "Z"]
    sliders = []
    checkboxes = []

    for i, axis in enumerate(axes):
        layout, slider, value_label, checkbox = _create_slider(
            axis, min_vals[i], max_vals[i], axis
        )
        slice_layout.addLayout(layout)
        sliders.append(slider)
        checkboxes.append(checkbox)

    return (
        slice_widget,
        (sliders[0], sliders[1], sliders[2]),
        (checkboxes[0], checkboxes[1], checkboxes[2]),
    )


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
    QWidget
        The right panel widget.
    """
    right_panel_widget = QWidget()
    right_layout = QVBoxLayout(right_panel_widget)
    right_layout.setContentsMargins(0, 10, 10, 10)
    right_layout.setSpacing(10)

    toggle_button_layout = QHBoxLayout()
    toggle_button_layout.addStretch(1)
    toggle_button_layout.addWidget(QPushButton("3D VIEW"))
    toggle_button_layout.addWidget(QPushButton("2D VIEW"))
    right_layout.addLayout(toggle_button_layout)

    right_layout.addWidget(viz_window)

    return right_panel_widget


def create_ui(viz_window):
    """Create the main UI layout with left and right panels.

    Parameters
    ----------
    viz_window : QWidget
        The main visualization window.

    Returns
    -------
    tuple
        A tuple containing the main widget, left panel, and right panel.
    """
    main_widget = QWidget()
    main_layout = QHBoxLayout(main_widget)

    left_panel = _create_left_panel()
    right_panel = _create_right_panel(viz_window)

    main_layout.addWidget(left_panel)
    main_layout.addWidget(right_panel, 1)

    return main_widget, left_panel, right_panel
