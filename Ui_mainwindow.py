# -*- coding: utf-8 -*-

#
# Created: Fri Nov 14 12:36:27 2014
#      by: pyside-uic 0.2.13 running on PySide 1.1.0
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1460, 739)
        MainWindow.setUnifiedTitleAndToolBarOnMac(True)
        self.centralWidget = QtGui.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")
        self.horizontalLayout_13 = QtGui.QHBoxLayout(self.centralWidget)
        self.horizontalLayout_13.setObjectName("horizontalLayout_13")
        self.verticalLayout_2 = QtGui.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.treeObject = QtGui.QTreeWidget(self.centralWidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.treeObject.sizePolicy().hasHeightForWidth())
        self.treeObject.setSizePolicy(sizePolicy)
        self.treeObject.setObjectName("treeObject")
        self.verticalLayout_2.addWidget(self.treeObject)
        self.tabProps_4 = QtGui.QTabWidget(self.centralWidget)
        self.tabProps_4.setEnabled(True)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tabProps_4.sizePolicy().hasHeightForWidth())
        self.tabProps_4.setSizePolicy(sizePolicy)
        self.tabProps_4.setObjectName("tabProps_4")
        self.tabPropsTract = QtGui.QWidget()
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tabPropsTract.sizePolicy().hasHeightForWidth())
        self.tabPropsTract.setSizePolicy(sizePolicy)
        self.tabPropsTract.setFocusPolicy(QtCore.Qt.TabFocus)
        self.tabPropsTract.setObjectName("tabPropsTract")
        self.horizontalLayout_2 = QtGui.QHBoxLayout(self.tabPropsTract)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.tblTract = QtGui.QTableWidget(self.tabPropsTract)
        self.tblTract.setEnabled(False)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tblTract.sizePolicy().hasHeightForWidth())
        self.tblTract.setSizePolicy(sizePolicy)
        self.tblTract.setAutoFillBackground(True)
        self.tblTract.setDragEnabled(False)
        self.tblTract.setShowGrid(True)
        self.tblTract.setGridStyle(QtCore.Qt.NoPen)
        self.tblTract.setRowCount(6)
        self.tblTract.setColumnCount(2)
        self.tblTract.setObjectName("tblTract")
        self.tblTract.setColumnCount(2)
        self.tblTract.setRowCount(6)
        item = QtGui.QTableWidgetItem()
        self.tblTract.setHorizontalHeaderItem(0, item)
        item = QtGui.QTableWidgetItem()
        self.tblTract.setHorizontalHeaderItem(1, item)
        brush = QtGui.QBrush(QtGui.QColor(221, 207, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        item = QtGui.QTableWidgetItem()
        item.setFlags(QtCore.Qt.ItemIsEnabled)
        item.setBackground(brush)
        self.tblTract.setItem(0, 0, item)
        brush = QtGui.QBrush(QtGui.QColor(221, 207, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        item = QtGui.QTableWidgetItem()
        item.setFlags(QtCore.Qt.ItemIsEnabled)
        item.setBackground(brush)
        self.tblTract.setItem(0, 1, item)
        brush = QtGui.QBrush(QtGui.QColor(216, 189, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        item = QtGui.QTableWidgetItem()
        item.setFlags(QtCore.Qt.ItemIsEnabled)
        item.setBackground(brush)
        self.tblTract.setItem(1, 0, item)
        brush = QtGui.QBrush(QtGui.QColor(216, 189, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        item = QtGui.QTableWidgetItem()
        item.setFlags(QtCore.Qt.ItemIsEnabled)
        item.setBackground(brush)
        self.tblTract.setItem(1, 1, item)
        brush = QtGui.QBrush(QtGui.QColor(221, 207, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        item = QtGui.QTableWidgetItem()
        item.setFlags(QtCore.Qt.ItemIsEnabled)
        item.setBackground(brush)
        self.tblTract.setItem(2, 0, item)
        brush = QtGui.QBrush(QtGui.QColor(221, 207, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        item = QtGui.QTableWidgetItem()
        item.setFlags(QtCore.Qt.ItemIsEnabled)
        item.setBackground(brush)
        self.tblTract.setItem(2, 1, item)
        brush = QtGui.QBrush(QtGui.QColor(216, 189, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        item = QtGui.QTableWidgetItem()
        item.setFlags(QtCore.Qt.ItemIsEnabled)
        item.setBackground(brush)
        self.tblTract.setItem(3, 0, item)
        brush = QtGui.QBrush(QtGui.QColor(216, 189, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        item = QtGui.QTableWidgetItem()
        item.setFlags(QtCore.Qt.ItemIsEnabled)
        item.setBackground(brush)
        self.tblTract.setItem(3, 1, item)
        brush = QtGui.QBrush(QtGui.QColor(221, 207, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        item = QtGui.QTableWidgetItem()
        item.setBackground(brush)
        self.tblTract.setItem(4, 0, item)
        brush = QtGui.QBrush(QtGui.QColor(221, 207, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        item = QtGui.QTableWidgetItem()
        item.setFlags(QtCore.Qt.ItemIsEnabled)
        item.setBackground(brush)
        self.tblTract.setItem(4, 1, item)
        brush = QtGui.QBrush(QtGui.QColor(216, 189, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        item = QtGui.QTableWidgetItem()
        item.setBackground(brush)
        self.tblTract.setItem(5, 0, item)
        brush = QtGui.QBrush(QtGui.QColor(216, 189, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        item = QtGui.QTableWidgetItem()
        item.setBackground(brush)
        self.tblTract.setItem(5, 1, item)
        self.tblTract.horizontalHeader().setVisible(False)
        self.tblTract.horizontalHeader().setCascadingSectionResizes(False)
        self.tblTract.horizontalHeader().setDefaultSectionSize(155)
        self.tblTract.horizontalHeader().setMinimumSectionSize(56)
        self.tblTract.horizontalHeader().setStretchLastSection(True)
        self.tblTract.verticalHeader().setVisible(False)
        self.tblTract.verticalHeader().setCascadingSectionResizes(False)
        self.tblTract.verticalHeader().setMinimumSectionSize(15)
        self.tblTract.verticalHeader().setStretchLastSection(True)
        self.horizontalLayout_2.addWidget(self.tblTract)
        self.tabProps_4.addTab(self.tabPropsTract, "")
        self.tabPropsROI = QtGui.QWidget()
        self.tabPropsROI.setEnabled(True)
        self.tabPropsROI.setObjectName("tabPropsROI")
        self.horizontalLayout_3 = QtGui.QHBoxLayout(self.tabPropsROI)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.tblROI = QtGui.QTableWidget(self.tabPropsROI)
        self.tblROI.setEnabled(False)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tblROI.sizePolicy().hasHeightForWidth())
        self.tblROI.setSizePolicy(sizePolicy)
        self.tblROI.setAutoFillBackground(True)
        self.tblROI.setDragEnabled(False)
        self.tblROI.setDragDropOverwriteMode(False)
        self.tblROI.setShowGrid(True)
        self.tblROI.setGridStyle(QtCore.Qt.NoPen)
        self.tblROI.setRowCount(7)
        self.tblROI.setColumnCount(2)
        self.tblROI.setObjectName("tblROI")
        self.tblROI.setColumnCount(2)
        self.tblROI.setRowCount(7)
        item = QtGui.QTableWidgetItem()
        self.tblROI.setVerticalHeaderItem(0, item)
        item = QtGui.QTableWidgetItem()
        self.tblROI.setHorizontalHeaderItem(0, item)
        item = QtGui.QTableWidgetItem()
        self.tblROI.setHorizontalHeaderItem(1, item)
        item = QtGui.QTableWidgetItem()
        item.setFlags(QtCore.Qt.ItemIsEnabled)
        self.tblROI.setItem(0, 0, item)
        item = QtGui.QTableWidgetItem()
        item.setFlags(QtCore.Qt.ItemIsEditable|QtCore.Qt.ItemIsEnabled)
        self.tblROI.setItem(0, 1, item)
        item = QtGui.QTableWidgetItem()
        item.setFlags(QtCore.Qt.ItemIsEnabled)
        self.tblROI.setItem(1, 0, item)
        item = QtGui.QTableWidgetItem()
        item.setFlags(QtCore.Qt.ItemIsEnabled)
        self.tblROI.setItem(1, 1, item)
        item = QtGui.QTableWidgetItem()
        item.setFlags(QtCore.Qt.ItemIsEnabled)
        self.tblROI.setItem(2, 0, item)
        item = QtGui.QTableWidgetItem()
        item.setFlags(QtCore.Qt.ItemIsEnabled)
        self.tblROI.setItem(2, 1, item)
        item = QtGui.QTableWidgetItem()
        item.setFlags(QtCore.Qt.ItemIsEnabled)
        self.tblROI.setItem(3, 0, item)
        item = QtGui.QTableWidgetItem()
        item.setFlags(QtCore.Qt.ItemIsEnabled)
        self.tblROI.setItem(3, 1, item)
        item = QtGui.QTableWidgetItem()
        item.setFlags(QtCore.Qt.ItemIsEnabled)
        self.tblROI.setItem(4, 0, item)
        item = QtGui.QTableWidgetItem()
        item.setFlags(QtCore.Qt.ItemIsEnabled)
        self.tblROI.setItem(4, 1, item)
        item = QtGui.QTableWidgetItem()
        item.setFlags(QtCore.Qt.ItemIsEnabled)
        self.tblROI.setItem(5, 0, item)
        item = QtGui.QTableWidgetItem()
        item.setFlags(QtCore.Qt.ItemIsEnabled)
        self.tblROI.setItem(5, 1, item)
        item = QtGui.QTableWidgetItem()
        item.setFlags(QtCore.Qt.ItemIsEnabled)
        self.tblROI.setItem(6, 0, item)
        item = QtGui.QTableWidgetItem()
        item.setFlags(QtCore.Qt.ItemIsEnabled)
        self.tblROI.setItem(6, 1, item)
        self.tblROI.horizontalHeader().setVisible(False)
        self.tblROI.horizontalHeader().setCascadingSectionResizes(False)
        self.tblROI.horizontalHeader().setDefaultSectionSize(135)
        self.tblROI.horizontalHeader().setStretchLastSection(True)
        self.tblROI.verticalHeader().setVisible(False)
        self.tblROI.verticalHeader().setCascadingSectionResizes(False)
        self.tblROI.verticalHeader().setStretchLastSection(True)
        self.horizontalLayout_3.addWidget(self.tblROI)
        self.tabProps_4.addTab(self.tabPropsROI, "")
        self.tab = QtGui.QWidget()
        self.tab.setObjectName("tab")
        self.horizontalLayout_4 = QtGui.QHBoxLayout(self.tab)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.tblROISlist = QtGui.QTableWidget(self.tab)
        self.tblROISlist.setEnabled(False)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tblROISlist.sizePolicy().hasHeightForWidth())
        self.tblROISlist.setSizePolicy(sizePolicy)
        self.tblROISlist.setAutoFillBackground(False)
        self.tblROISlist.setShowGrid(False)
        self.tblROISlist.setColumnCount(1)
        self.tblROISlist.setObjectName("tblROISlist")
        self.tblROISlist.setColumnCount(1)
        self.tblROISlist.setRowCount(0)
        self.tblROISlist.horizontalHeader().setVisible(False)
        self.tblROISlist.horizontalHeader().setDefaultSectionSize(624)
        self.tblROISlist.horizontalHeader().setHighlightSections(False)
        self.tblROISlist.horizontalHeader().setMinimumSectionSize(624)
        self.tblROISlist.verticalHeader().setVisible(False)
        self.tblROISlist.verticalHeader().setHighlightSections(False)
        self.horizontalLayout_4.addWidget(self.tblROISlist)
        self.tabProps_4.addTab(self.tab, "")
        self.verticalLayout_2.addWidget(self.tabProps_4)
        self.horizontalLayout_28 = QtGui.QHBoxLayout()
        self.horizontalLayout_28.setObjectName("horizontalLayout_28")
        self.chkbShowTract = QtGui.QCheckBox(self.centralWidget)
        self.chkbShowTract.setEnabled(False)
        self.chkbShowTract.setChecked(True)
        self.chkbShowTract.setTristate(False)
        self.chkbShowTract.setObjectName("chkbShowTract")
        self.horizontalLayout_28.addWidget(self.chkbShowTract)
        self.chkbShowStruct = QtGui.QCheckBox(self.centralWidget)
        self.chkbShowStruct.setEnabled(False)
        self.chkbShowStruct.setChecked(True)
        self.chkbShowStruct.setObjectName("chkbShowStruct")
        self.horizontalLayout_28.addWidget(self.chkbShowStruct)
        self.verticalLayout_2.addLayout(self.horizontalLayout_28)
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.grbROImethod = QtGui.QGroupBox(self.centralWidget)
        self.grbROImethod.setEnabled(False)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.grbROImethod.sizePolicy().hasHeightForWidth())
        self.grbROImethod.setSizePolicy(sizePolicy)
        self.grbROImethod.setAutoFillBackground(True)
        self.grbROImethod.setFlat(False)
        self.grbROImethod.setObjectName("grbROImethod")
        self.verticalLayout_3 = QtGui.QVBoxLayout(self.grbROImethod)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout_31 = QtGui.QHBoxLayout()
        self.horizontalLayout_31.setObjectName("horizontalLayout_31")
        self.rdbInsSphere = QtGui.QRadioButton(self.grbROImethod)
        self.rdbInsSphere.setChecked(True)
        self.rdbInsSphere.setObjectName("rdbInsSphere")
        self.horizontalLayout_31.addWidget(self.rdbInsSphere)
        self.rdbtrackvis = QtGui.QRadioButton(self.grbROImethod)
        self.rdbtrackvis.setObjectName("rdbtrackvis")
        self.horizontalLayout_31.addWidget(self.rdbtrackvis)
        self.verticalLayout_3.addLayout(self.horizontalLayout_31)
        self.verticalLayout.addWidget(self.grbROImethod)
        self.grbCluster = QtGui.QGroupBox(self.centralWidget)
        self.grbCluster.setEnabled(False)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.grbCluster.sizePolicy().hasHeightForWidth())
        self.grbCluster.setSizePolicy(sizePolicy)
        self.grbCluster.setObjectName("grbCluster")
        self.horizontalLayout_32 = QtGui.QHBoxLayout(self.grbCluster)
        self.horizontalLayout_32.setObjectName("horizontalLayout_32")
        self.horizontalLayout_33 = QtGui.QHBoxLayout()
        self.horizontalLayout_33.setObjectName("horizontalLayout_33")
        self.spbRecluster = QtGui.QSpinBox(self.grbCluster)
        self.spbRecluster.setReadOnly(False)
        self.spbRecluster.setObjectName("spbRecluster")
        self.horizontalLayout_33.addWidget(self.spbRecluster)
        self.hSlReCluster = QtGui.QSlider(self.grbCluster)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.hSlReCluster.sizePolicy().hasHeightForWidth())
        self.hSlReCluster.setSizePolicy(sizePolicy)
        self.hSlReCluster.setMinimum(1)
        self.hSlReCluster.setMaximum(150)
        self.hSlReCluster.setOrientation(QtCore.Qt.Horizontal)
        self.hSlReCluster.setInvertedAppearance(False)
        self.hSlReCluster.setTickPosition(QtGui.QSlider.NoTicks)
        self.hSlReCluster.setTickInterval(0)
        self.hSlReCluster.setObjectName("hSlReCluster")
        self.horizontalLayout_33.addWidget(self.hSlReCluster)
        self.horizontalLayout_34 = QtGui.QHBoxLayout()
        self.horizontalLayout_34.setObjectName("horizontalLayout_34")
        spacerItem = QtGui.QSpacerItem(18, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_34.addItem(spacerItem)
        self.pbRecluster = QtGui.QPushButton(self.grbCluster)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pbRecluster.sizePolicy().hasHeightForWidth())
        self.pbRecluster.setSizePolicy(sizePolicy)
        self.pbRecluster.setObjectName("pbRecluster")
        self.horizontalLayout_34.addWidget(self.pbRecluster)
        spacerItem1 = QtGui.QSpacerItem(17, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_34.addItem(spacerItem1)
        self.horizontalLayout_33.addLayout(self.horizontalLayout_34)
        self.horizontalLayout_32.addLayout(self.horizontalLayout_33)
        self.verticalLayout.addWidget(self.grbCluster)
        self.grbExtendcluster = QtGui.QGroupBox(self.centralWidget)
        self.grbExtendcluster.setEnabled(False)
        self.grbExtendcluster.setObjectName("grbExtendcluster")
        self.horizontalLayout_12 = QtGui.QHBoxLayout(self.grbExtendcluster)
        self.horizontalLayout_12.setObjectName("horizontalLayout_12")
        self.horizontalLayout_5 = QtGui.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.spbExtClust = QtGui.QSpinBox(self.grbExtendcluster)
        self.spbExtClust.setReadOnly(False)
        self.spbExtClust.setMinimum(0)
        self.spbExtClust.setObjectName("spbExtClust")
        self.horizontalLayout_5.addWidget(self.spbExtClust)
        self.hSlExtclust = QtGui.QSlider(self.grbExtendcluster)
        self.hSlExtclust.setMinimum(0)
        self.hSlExtclust.setOrientation(QtCore.Qt.Horizontal)
        self.hSlExtclust.setObjectName("hSlExtclust")
        self.horizontalLayout_5.addWidget(self.hSlExtclust)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem2 = QtGui.QSpacerItem(13, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem2)
        self.pbExtCluster = QtGui.QPushButton(self.grbExtendcluster)
        self.pbExtCluster.setObjectName("pbExtCluster")
        self.horizontalLayout.addWidget(self.pbExtCluster)
        spacerItem3 = QtGui.QSpacerItem(13, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem3)
        self.horizontalLayout_5.addLayout(self.horizontalLayout)
        self.horizontalLayout_12.addLayout(self.horizontalLayout_5)
        self.verticalLayout.addWidget(self.grbExtendcluster)
        self.verticalLayout_2.addLayout(self.verticalLayout)
        self.horizontalLayout_13.addLayout(self.verticalLayout_2)
        self.gridWidget_4 = QtGui.QWidget(self.centralWidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.gridWidget_4.sizePolicy().hasHeightForWidth())
        self.gridWidget_4.setSizePolicy(sizePolicy)
        self.gridWidget_4.setObjectName("gridWidget_4")
        self.gridLayout_4 = QtGui.QGridLayout(self.gridWidget_4)
        self.gridLayout_4.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.horizontalLayout_13.addWidget(self.gridWidget_4)
        MainWindow.setCentralWidget(self.centralWidget)
        self.menuBar = QtGui.QMenuBar(MainWindow)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 1460, 25))
        self.menuBar.setNativeMenuBar(False)
        self.menuBar.setObjectName("menuBar")
        self.menuFile = QtGui.QMenu(self.menuBar)
        self.menuFile.setObjectName("menuFile")
        self.menuROI = QtGui.QMenu(self.menuBar)
        self.menuROI.setEnabled(False)
        self.menuROI.setObjectName("menuROI")
        MainWindow.setMenuBar(self.menuBar)
        self.actionLoad_Structural_Image = QtGui.QAction(MainWindow)
        self.actionLoad_Structural_Image.setCheckable(False)
        self.actionLoad_Structural_Image.setObjectName("actionLoad_Structural_Image")
        self.actionLoad_Tractography = QtGui.QAction(MainWindow)
        self.actionLoad_Tractography.setEnabled(False)
        self.actionLoad_Tractography.setObjectName("actionLoad_Tractography")
        self.actionLoad_Saved_Segmentation = QtGui.QAction(MainWindow)
        self.actionLoad_Saved_Segmentation.setObjectName("actionLoad_Saved_Segmentation")
        self.actionSave_Segmentation = QtGui.QAction(MainWindow)
        self.actionSave_Segmentation.setEnabled(False)
        self.actionSave_Segmentation.setObjectName("actionSave_Segmentation")
        self.actionClose = QtGui.QAction(MainWindow)
        self.actionClose.setObjectName("actionClose")
        self.actionLoad_MASk = QtGui.QAction(MainWindow)
        self.actionLoad_MASk.setEnabled(True)
        self.actionLoad_MASk.setObjectName("actionLoad_MASk")
        self.actionCreate_Sphere = QtGui.QAction(MainWindow)
        self.actionCreate_Sphere.setObjectName("actionCreate_Sphere")
        self.actionExpand_ROI = QtGui.QAction(MainWindow)
        self.actionExpand_ROI.setObjectName("actionExpand_ROI")
        self.actionSegment_by_Clustering = QtGui.QAction(MainWindow)
        self.actionSegment_by_Clustering.setObjectName("actionSegment_by_Clustering")
        self.actionRe_Cluster = QtGui.QAction(MainWindow)
        self.actionRe_Cluster.setObjectName("actionRe_Cluster")
        self.actionExpand_Clusters = QtGui.QAction(MainWindow)
        self.actionExpand_Clusters.setObjectName("actionExpand_Clusters")
        self.actionSave_as_trackvis_file = QtGui.QAction(MainWindow)
        self.actionSave_as_trackvis_file.setEnabled(False)
        self.actionSave_as_trackvis_file.setObjectName("actionSave_as_trackvis_file")
        self.menuFile.addAction(self.actionLoad_Structural_Image)
        self.menuFile.addAction(self.actionLoad_Tractography)
        self.menuFile.addAction(self.actionLoad_Saved_Segmentation)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionSave_Segmentation)
        self.menuFile.addAction(self.actionSave_as_trackvis_file)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionClose)
        self.menuROI.addAction(self.actionLoad_MASk)
        self.menuROI.addAction(self.actionCreate_Sphere)
        self.menuBar.addAction(self.menuFile.menuAction())
        self.menuBar.addAction(self.menuROI.menuAction())

        self.retranslateUi(MainWindow)
        self.tabProps_4.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QtGui.QApplication.translate("MainWindow", "Tractome", None, QtGui.QApplication.UnicodeUTF8))
        self.treeObject.headerItem().setText(0, QtGui.QApplication.translate("MainWindow", "Objects", None, QtGui.QApplication.UnicodeUTF8))
        self.tblTract.setSortingEnabled(False)
        self.tblTract.horizontalHeaderItem(0).setText(QtGui.QApplication.translate("MainWindow", "Properties", None, QtGui.QApplication.UnicodeUTF8))
        self.tblTract.horizontalHeaderItem(1).setText(QtGui.QApplication.translate("MainWindow", "Values", None, QtGui.QApplication.UnicodeUTF8))
        __sortingEnabled = self.tblTract.isSortingEnabled()
        self.tblTract.setSortingEnabled(False)
        self.tblTract.item(0, 0).setText(QtGui.QApplication.translate("MainWindow", "Name", None, QtGui.QApplication.UnicodeUTF8))
        self.tblTract.item(1, 0).setText(QtGui.QApplication.translate("MainWindow", "Track Count", None, QtGui.QApplication.UnicodeUTF8))
        self.tblTract.item(2, 0).setText(QtGui.QApplication.translate("MainWindow", "Representative Count", None, QtGui.QApplication.UnicodeUTF8))
        self.tblTract.item(3, 0).setText(QtGui.QApplication.translate("MainWindow", "Voxel Size", None, QtGui.QApplication.UnicodeUTF8))
        self.tblTract.item(4, 0).setText(QtGui.QApplication.translate("MainWindow", "Volume Size", None, QtGui.QApplication.UnicodeUTF8))
        self.tblTract.item(5, 0).setText(QtGui.QApplication.translate("MainWindow", "Orientation", None, QtGui.QApplication.UnicodeUTF8))
        self.tblTract.setSortingEnabled(__sortingEnabled)
        self.tabProps_4.setTabText(self.tabProps_4.indexOf(self.tabPropsTract), QtGui.QApplication.translate("MainWindow", "Tractography", None, QtGui.QApplication.UnicodeUTF8))
        self.tblROI.setSortingEnabled(False)
        self.tblROI.verticalHeaderItem(0).setText(QtGui.QApplication.translate("MainWindow", "1", None, QtGui.QApplication.UnicodeUTF8))
        self.tblROI.horizontalHeaderItem(0).setText(QtGui.QApplication.translate("MainWindow", "Properties", None, QtGui.QApplication.UnicodeUTF8))
        self.tblROI.horizontalHeaderItem(1).setText(QtGui.QApplication.translate("MainWindow", "Values", None, QtGui.QApplication.UnicodeUTF8))
        __sortingEnabled = self.tblROI.isSortingEnabled()
        self.tblROI.setSortingEnabled(False)
        self.tblROI.item(0, 0).setText(QtGui.QApplication.translate("MainWindow", "Name", None, QtGui.QApplication.UnicodeUTF8))
        self.tblROI.item(1, 0).setText(QtGui.QApplication.translate("MainWindow", "Visible", None, QtGui.QApplication.UnicodeUTF8))
        self.tblROI.item(2, 0).setText(QtGui.QApplication.translate("MainWindow", "Color", None, QtGui.QApplication.UnicodeUTF8))
        self.tblROI.item(3, 0).setText(QtGui.QApplication.translate("MainWindow", "X coordinate", None, QtGui.QApplication.UnicodeUTF8))
        self.tblROI.item(4, 0).setText(QtGui.QApplication.translate("MainWindow", "Y coordinate", None, QtGui.QApplication.UnicodeUTF8))
        self.tblROI.item(5, 0).setText(QtGui.QApplication.translate("MainWindow", "Z coordinate", None, QtGui.QApplication.UnicodeUTF8))
        self.tblROI.item(6, 0).setText(QtGui.QApplication.translate("MainWindow", "Radius", None, QtGui.QApplication.UnicodeUTF8))
        self.tblROI.setSortingEnabled(__sortingEnabled)
        self.tabProps_4.setTabText(self.tabProps_4.indexOf(self.tabPropsROI), QtGui.QApplication.translate("MainWindow", "ROI Properties", None, QtGui.QApplication.UnicodeUTF8))
        self.tabProps_4.setTabText(self.tabProps_4.indexOf(self.tab), QtGui.QApplication.translate("MainWindow", "Applied ROIS", None, QtGui.QApplication.UnicodeUTF8))
        self.chkbShowTract.setText(QtGui.QApplication.translate("MainWindow", "Show Tractography", None, QtGui.QApplication.UnicodeUTF8))
        self.chkbShowStruct.setText(QtGui.QApplication.translate("MainWindow", "Show Structural Image", None, QtGui.QApplication.UnicodeUTF8))
        self.grbROImethod.setTitle(QtGui.QApplication.translate("MainWindow", "ROI Method", None, QtGui.QApplication.UnicodeUTF8))
        self.rdbInsSphere.setText(QtGui.QApplication.translate("MainWindow", "Inside Sphere", None, QtGui.QApplication.UnicodeUTF8))
        self.rdbtrackvis.setText(QtGui.QApplication.translate("MainWindow", "Trackvis Like", None, QtGui.QApplication.UnicodeUTF8))
        self.grbCluster.setTitle(QtGui.QApplication.translate("MainWindow", "Re-Cluster", None, QtGui.QApplication.UnicodeUTF8))
        self.pbRecluster.setText(QtGui.QApplication.translate("MainWindow", "Apply", None, QtGui.QApplication.UnicodeUTF8))
        self.grbExtendcluster.setTitle(QtGui.QApplication.translate("MainWindow", "Extend Clusters", None, QtGui.QApplication.UnicodeUTF8))
        self.pbExtCluster.setText(QtGui.QApplication.translate("MainWindow", "Set Clusters", None, QtGui.QApplication.UnicodeUTF8))
        self.menuFile.setTitle(QtGui.QApplication.translate("MainWindow", "File", None, QtGui.QApplication.UnicodeUTF8))
        self.menuROI.setTitle(QtGui.QApplication.translate("MainWindow", "ROI", None, QtGui.QApplication.UnicodeUTF8))
        self.actionLoad_Structural_Image.setText(QtGui.QApplication.translate("MainWindow", "Load Structural Image", None, QtGui.QApplication.UnicodeUTF8))
        self.actionLoad_Tractography.setText(QtGui.QApplication.translate("MainWindow", "Load Tractography", None, QtGui.QApplication.UnicodeUTF8))
        self.actionLoad_Saved_Segmentation.setText(QtGui.QApplication.translate("MainWindow", "Load Segmentation", None, QtGui.QApplication.UnicodeUTF8))
        self.actionSave_Segmentation.setText(QtGui.QApplication.translate("MainWindow", "Save Segmentation", None, QtGui.QApplication.UnicodeUTF8))
        self.actionClose.setText(QtGui.QApplication.translate("MainWindow", "Close", None, QtGui.QApplication.UnicodeUTF8))
        self.actionLoad_MASk.setText(QtGui.QApplication.translate("MainWindow", "Load Mask", None, QtGui.QApplication.UnicodeUTF8))
        self.actionCreate_Sphere.setText(QtGui.QApplication.translate("MainWindow", "Create Sphere", None, QtGui.QApplication.UnicodeUTF8))
        self.actionExpand_ROI.setText(QtGui.QApplication.translate("MainWindow", "Expand ROI", None, QtGui.QApplication.UnicodeUTF8))
        self.actionSegment_by_Clustering.setText(QtGui.QApplication.translate("MainWindow", "Segment by Clustering", None, QtGui.QApplication.UnicodeUTF8))
        self.actionRe_Cluster.setText(QtGui.QApplication.translate("MainWindow", "Re-Cluster", None, QtGui.QApplication.UnicodeUTF8))
        self.actionExpand_Clusters.setText(QtGui.QApplication.translate("MainWindow", "Expand Clusters", None, QtGui.QApplication.UnicodeUTF8))
        self.actionSave_as_trackvis_file.setText(QtGui.QApplication.translate("MainWindow", "Save as trackvis file", None, QtGui.QApplication.UnicodeUTF8))

