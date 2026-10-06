"""Reusable modal dialogs for the tractome app."""

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QVBoxLayout,
)

from tractome.io import get_embedding_label
from tractome.ui._paths import IMAGES_PATH


def _build_brand_header(layout):
    """Prepend a compact Tractome logo + title header to a dialog layout.

    Parameters
    ----------
    layout : QVBoxLayout
        The dialog's top-level layout, still empty; the header and a
        separator line are added first so later content follows below.
    """
    header_row = QHBoxLayout()
    header_row.setContentsMargins(0, 0, 0, 0)
    header_row.setSpacing(8)

    logo_label = QLabel()
    logo_pixmap = QPixmap(str(IMAGES_PATH / "logo.png"))
    logo_label.setPixmap(
        logo_pixmap.scaled(64, 18, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    )
    header_row.addWidget(logo_label)

    title_label = QLabel("Tractome")
    title_label.setObjectName("dialogBrandTitle")
    header_row.addWidget(title_label)
    header_row.addStretch()
    layout.addLayout(header_row)

    separator = QFrame()
    separator.setObjectName("dialogBrandSeparator")
    separator.setFrameShape(QFrame.HLine)
    layout.addWidget(separator)


class CreditsDialog(QDialog):
    """Modal "about" dialog with the Tractome logo, version, and credits."""

    def __init__(self, parent=None):
        """Build the credits dialog.

        Parameters
        ----------
        parent : QWidget, optional
            The parent widget.
        """
        super().__init__(parent)
        self.setObjectName("creditsDialog")
        self.setWindowTitle("About Tractome")
        self.setModal(True)
        self.setFixedWidth(520)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(4)
        layout.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        logo_label = QLabel()
        logo_pixmap = QPixmap(str(IMAGES_PATH / "logo.png"))
        logo_label.setPixmap(
            logo_pixmap.scaled(127, 35, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        layout.addWidget(logo_label)

        title_label = QLabel("Tractome")
        title_label.setObjectName("creditsTitle")
        layout.addWidget(title_label)

        version_label = QLabel("Version 2.0.0a1 - June 2026")
        version_label.setObjectName("creditsVersion")
        layout.addWidget(version_label)

        layout.addSpacing(16)

        for text in (
            "Neuroinformatics Lab (NILab), Fondazione Bruno Kessler",
            "GRG, Indiana University",
        ):
            label = QLabel(text)
            label.setObjectName("creditsBody")
            layout.addWidget(label)

        layout.addSpacing(12)

        url_label = QLabel("https://tractome.org")
        url_label.setObjectName("creditsLink")
        layout.addWidget(url_label)

        support_label = QLabel("Support: help@tractome.org")
        support_label.setObjectName("creditsLink")
        layout.addWidget(support_label)

        layout.addSpacing(30)

        references_header = QLabel("REFERENCES")
        references_header.setObjectName("creditsSectionHeader")
        layout.addWidget(references_header)

        for reference in (
            "Sarubbo S, et al. (2024) Changing the Paradigm for Tractography "
            "Segmentation in Neurosurgery: Validation of a Streamline-Based "
            "Approach, Brain Sciences, 14(12) doi:10.3390/brainsci14121232",
            "Porro-Munoz D. et al. (2015) Tractome: a visual data mining tool "
            "for brain connectivity analysis, Data mining and Knowledge "
            "Discovery, 29(5) doi:10.1007/s10618-015-0408-z",
        ):
            label = QLabel(reference)
            label.setObjectName("creditsReference")
            label.setWordWrap(True)
            layout.addWidget(label)

        layout.addSpacing(16)

        funding_label = QLabel(
            "This work was supported by FAIR Foundation, VRT Foundation, "
            "FBK Foundation"
        )
        funding_label.setObjectName("creditsBody")
        funding_label.setWordWrap(True)
        layout.addWidget(funding_label)


class EmbeddingSelectionDialog(QDialog):
    """Ask the user which embedding to use for clustering.

    Presented when a tractogram ships with more than one embedding. Each
    available embedding is offered as a radio button. The button text is the
    embedding's user-facing label (e.g. the stored key ``"dismatrix"`` is
    shown as ``"dissimilarity"``), while the value returned is the underlying
    stored key.
    """

    def __init__(self, embedding_names, parent=None):
        """Build the selection dialog.

        Parameters
        ----------
        embedding_names : list[str]
            Stored ``data_per_streamline`` keys of the embeddings available
            on the tractogram.
        parent : QWidget, optional
            The parent widget.
        """
        super().__init__(parent)
        self.setObjectName("embeddingSelectionDialog")
        self.setWindowTitle("Select embedding")
        self.setModal(True)

        layout = QVBoxLayout(self)
        _build_brand_header(layout)
        layout.addWidget(
            QLabel(
                "This tractogram contains multiple embeddings.\n"
                "Choose which one to use for clustering:"
            )
        )

        self._button_group = QButtonGroup(self)
        for index, name in enumerate(embedding_names):
            radio = QRadioButton(get_embedding_label(name))
            if index == 0:
                radio.setChecked(True)
            self._button_group.addButton(radio, index)
            layout.addWidget(radio)

        self._embedding_names = list(embedding_names)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @property
    def selected_embedding(self):
        """Return the embedding name chosen by the user.

        Returns
        -------
        str or None
            The selected embedding name, or None if nothing is selected.
        """
        checked_id = self._button_group.checkedId()
        if checked_id < 0:
            return None
        return self._embedding_names[checked_id]
