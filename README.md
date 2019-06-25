Tractome
========

Tractome (http://tractome.org) is an interactive clustering-based 3D tool for exploration and segmentation of tractography data.

Tractome is an interactive tool for visualisation, exploration and segmentation of tractography data. It supports neuroanatomists and medical doctors in their study of white matter anatomical structures from diffusion magnetic resonance imaging (dMRI) data. Unlike previous systems for tractography segmentation, Tractome is a computer-assisted tool in which the user interacts with a summary of the tractography instead of the whole set of streamlines. The summary is generated by clustering the tractography into a desired number of clusters, usually in the order of tens or one hundred. The summary shows only one representative streamline for each cluster, so that it is easier to interact with them in the 3D scene. The user can then iteratively select the representative streamlines of interest and re-cluster the associated sets or streamlines into smaller ones in order to incrementally reveal details of the anatomical structure of interest. The interaction is powered by novel efficient algorithms for fast clustering.  Tractome is written in Python language and it supports the TrackVis format and the DiPy format for tractography files.

Example video: [tractome-video](https://youtu.be/159UAr9uVHk)

Download
------------

* Binary: [tractome-macos.dmg](https://bit.ly/dmg4tractome2) (Mac OS X 10.9.5 or later)
* Sources: [tractome.tar.gz](https://github.com/FBK-NILab/tractome/archive/master.tar.gz)

Dependencies
------------

* pyglet : http://www.pyglet.org , provides the Python bindings to OpenGL.
* fos : https://github.com/fos/fos.git , provides high-level OpenGL Actors for scientific visualisation
* PySide : http://qt-project.org/wiki/PySide , Python bindings for the Qt libraries.
* NiBabel : http://nipy.org/nibabel , provides read and write access to common medical and neuroimaging file formats.
* Dipy: http://www.dipy.org , provides tools for dMRI data analysis.
* scikit-learn : http://scikit-learn.org , provides the Mini-Batch K-means clustering algorithm.

The software is developed and tested on Ubuntu 12.04 LTS using the Neurodebian repositories. In Debian/Ubuntu systems the packages of the dependencies - with the exception of fos - can be installed with
```
apt-get install python-pyglet python-pyside python-nibabel python-dipy python-sklearn
```

<!-- With the IPyhon version shipped with Ubuntu 12.04 (IPython v0.12.1), there are issues for the proper performance/visualization of some Qt Dialogs, e.g. QColorDialog. This is already solved in a more recent version of IPython (v0.13.2), which can be directly installed from Ubuntu backports repositories with the following steps: -->

<!-- 1. Add Ubuntu backports repositories: "go to Software Sources, switch to the Updates tab and make sure Unsupported updates is checked". -->
<!-- 2. ```apt-get install ipython/precise-backports``` -->

<!-- For more details about Ubuntu backports: https://help.ubuntu.com/community/UbuntuBackports -->

To Run Tractome
---------------
```
python mainwindow.py
```

First of all load a structural image (```File -> Load Structural```). Then load a related tractography (```File -> Load Tractography```), either in TrackVis format or Dipy format. Tractome loads the file and then executes some pre-computations that may require some time - from seconds to a couple of minutes, depending on the size of the tractography. These pre-computations are saved in the same directory of the tractography, so the second time you load that tractography this step will be faster. After loading, Tractome shows the structural and tractography data in a 3D scene where standard operations like rotating, dragging and zooming are available.

Tractome shows clusters of the whole tractography in the 3D scene by displaying the medoid streamline, called *representative*, of each cluster. You can interact with the clusters in several ways through their representatives:

1. Point one representative streamline with the mouse pointer and then press **P** to select/pick the corresponding cluster. When selected, the representative becomes white. You can select as many representatives/clusters as you like.
2. Expand the selected clusters in order to show their streamlines by pressing **E**. Press **E** again to toggle expansion.
3. When you are happy with your selection then press **Backspace** to remove all the clusters and streamlines not selected.
4. You can re-cluster the streamlines by clicking the button **Apply** in the bottom-left corner after choosing the desired number of cluster with the slider nearby.
5. Repeat the previous steps as many time as you want. If you want to step-back one step (undo) press **B**. To move one step forward (redo) press **F**.
6. You can save the steps done so far with ```File -> Save Segmentation```.


Example Datasets
----------------

Two example datasets are available, with structural, tractography and pre-computed files:

1. [HCP_subject124422_100Kseeds.tgz](http://nilab.cimec.unitn.it/nilab/hcp/HCP_subject124422_100Kseeds.tgz) : 15k streamlines.
2. [HCP_subject124422_3Mseeds.tgz](http://nilab.cimec.unitn.it/nilab/hcp/HCP_subject124422_3Mseeds.tgz) : 460k streamlines (be careful, at least 4Gb RAM needed).

Those data refer to subject 124422 of the [Human Connectome Project](http://www.humanconnectome.org/). The tractography is reconstructed following [these steps](https://github.com/FBK-NILab/HCP-Tractography) with 100k seeds and 3 millions seeds respectively.
