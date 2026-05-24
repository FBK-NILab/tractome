import numpy as np

from tractome.io import read_csv, read_mesh, read_nifti, read_tractogram


def _as_filter_volume(volume):
    """Collapse an ROI volume to 3D so it can drive streamline filtering.

    NIfTI ROIs may arrive as ``(X, Y, Z, C)`` (e.g. RGB/RGBA labelmaps)
    or with a structured RGB dtype. The filter pipeline expects a
    plain ``(X, Y, Z)`` mask, so trailing channels are reduced with
    ``any``: a voxel is occupied if any channel is non-zero.

    Parameters
    ----------
    volume : ndarray
        ROI volume loaded from disk or created in memory.

    Returns
    -------
    ndarray
        A 3D ROI mask suitable for filtering.
    """
    arr = np.asarray(volume)
    if arr.dtype.names is not None:
        arr = np.stack([arr[name] for name in arr.dtype.names], axis=-1)
    if arr.ndim > 3:
        arr = np.any(arr != 0, axis=tuple(range(3, arr.ndim)))
    return arr


class InputManager:
    """Manage input file paths and track the currently active items.

    This manager stores tractogram, T1, mesh, mesh texture, ROI, and parcel
    file paths while keeping the index of the currently selected item for each
    input type.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        """Create a new instance of the InputManager if one does not exist.

        Parameters
        ----------
        *args : tuple
            Variable length argument list.
        **kwargs : dict
            Arbitrary keyword arguments.

        Returns
        -------
        InputManager
            The instance of the InputManager.
        """
        if not cls._instance:
            cls._instance = super(InputManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Manage input file paths and track the currently active items.

        This manager stores tractogram, T1, mesh, mesh texture, ROI, and
        parcel file paths while keeping the index of the currently selected
        item for each input type.
        """
        self._provided_inputs = {
            "tractogram": [],
            "t1": [],
            "mesh": [],
            "mesh_texture": [],
            "roi": [],
            "parcel": [],
        }

        self._current_inputs = {
            "tractogram": -1,
            "t1": -1,
            "mesh": -1,
            "mesh_texture": -1,
            "roi": -1,
            "parcel": -1,
        }

        self._loaded_inputs = {
            "tractogram": None,
            "t1": None,
            "mesh": None,
            "roi": None,
            "parcel": None,
        }
        self._loaded_rois = {}
        self._created_roi_counters = {}

    def reset(self):
        """Clear all input paths, selections, and cached loaded data."""
        self.__init__()

    def add_tractogram(self, tractogram):
        """Add a tractogram path and make it the current tractogram.

        Parameters
        ----------
        tractogram : str
            Path to the tractogram file to store.
        """
        self._provided_inputs["tractogram"].append(tractogram)
        self._current_inputs["tractogram"] = (
            len(self._provided_inputs["tractogram"]) - 1
        )

    def add_t1(self, t1):
        """Add a T1 image path and make it the current T1 image.

        Parameters
        ----------
        t1 : str
            Path to the T1 image file to store.
        """
        if t1 in self._provided_inputs["t1"]:
            self._current_inputs["t1"] = self._provided_inputs["t1"].index(t1)
            return

        self._provided_inputs["t1"].append(t1)
        self._current_inputs["t1"] = len(self._provided_inputs["t1"]) - 1

    def add_mesh(self, mesh, mesh_texture):
        """Add a mesh path and its texture path, and make them current.

        Parameters
        ----------
        mesh : str
            Path to the mesh file to store.
        mesh_texture : str
            Path to the texture file associated with ``mesh``.
        """
        self._provided_inputs["mesh"].append(mesh)
        self._provided_inputs["mesh_texture"].append(mesh_texture)
        idx = len(self._provided_inputs["mesh"]) - 1
        self._current_inputs["mesh"] = idx
        self._current_inputs["mesh_texture"] = idx
        self._loaded_inputs["mesh"] = None

    def add_roi(self, roi):
        """Add an ROI path and make it the current ROI.

        Parameters
        ----------
        roi : str
            Path to the ROI file to store.
        """
        self._provided_inputs["roi"].append(roi)
        self._current_inputs["roi"] = len(self._provided_inputs["roi"]) - 1

    def update_roi_volume(self, roi_id, volume, affine):
        """Replace the cached volume/affine for an existing in-memory ROI.

        Used by the interactive ROI create mode: while the user keeps
        re-dragging without committing, the same synthetic ROI entry is
        overwritten in place rather than spawning a new one.

        Parameters
        ----------
        roi_id : str
            Identifier returned by :meth:`add_roi_volume`.
        volume : ndarray
            Replacement binary ROI volume.
        affine : ndarray
            Replacement voxel-to-world affine.
        """
        if roi_id not in self._loaded_rois:
            raise ValueError(f"Unknown in-memory ROI: {roi_id!r}")
        index = self._provided_inputs["roi"].index(roi_id)
        self._loaded_rois[roi_id] = (volume, affine, roi_id, index)

    def add_roi_volume(self, volume, affine, *, label="roi"):
        """Add an ROI from an in-memory binary volume and make it current.

        The ROI is registered under a synthetic identifier of the form
        ``"<created>:<label>:<n>"``. The identifier takes the place of
        a file path in the input manager so per-ROI dicts keyed by path
        (colors, visibility, applied/negated flags) keep working.

        Parameters
        ----------
        volume : ndarray
            Binary ROI volume.
        affine : ndarray
            Voxel-to-world affine matrix shared with the reference image.
        label : str, optional
            Human-readable shape tag included in the synthetic id.

        Returns
        -------
        str
            The synthetic identifier registered in the input manager.
        """
        total = self._created_roi_counters.get("__total__", 0) + 1
        self._created_roi_counters["__total__"] = total
        roi_id = f"ROI {total}"
        self._provided_inputs["roi"].append(roi_id)
        index = len(self._provided_inputs["roi"]) - 1
        self._current_inputs["roi"] = index
        self._loaded_rois[roi_id] = (volume, affine, roi_id, index)
        return roi_id

    def add_parcel(self, parcel):
        """Add a parcel path and make it the current parcel.

        Parameters
        ----------
        parcel : str
            Path to the parcel file to store.
        """
        if parcel in self._provided_inputs["parcel"]:
            self._current_inputs["parcel"] = self._provided_inputs["parcel"].index(
                parcel
            )
            self._loaded_inputs["parcel"] = None
            return

        self._provided_inputs["parcel"].append(parcel)
        self._current_inputs["parcel"] = len(self._provided_inputs["parcel"]) - 1

    def get_current_tractogram(self):
        """Load and return the current tractogram.

        Returns
        -------
        tuple
            ``(sft, reference, path, index)`` for the current tractogram.

        Raises
        ------
        ValueError
            If no tractogram is available.
        """
        if self._current_inputs["tractogram"] == -1:
            raise ValueError("No tractogram available.")

        idx = self._current_inputs["tractogram"]
        if (
            self._loaded_inputs["tractogram"] is not None
            and self._loaded_inputs["tractogram"][3] == idx
        ):
            return self._loaded_inputs["tractogram"]

        path = self._provided_inputs["tractogram"][idx]
        reference = None
        if self._current_inputs["t1"] != -1:
            reference = self._provided_inputs["t1"][self._current_inputs["t1"]]
        sft = read_tractogram(path, reference=reference)
        self._loaded_inputs["tractogram"] = (sft, reference, path, idx)
        return self._loaded_inputs["tractogram"]

    def get_current_t1(self):
        """Load and return the current T1 image.

        Returns
        -------
        tuple
            ``(nifti_img, affine, path, index)`` for the current T1 image.

        Raises
        ------
        ValueError
            If no T1 image is available.
        """
        if self._current_inputs["t1"] == -1:
            raise ValueError("No T1 image available.")

        idx = self._current_inputs["t1"]
        if (
            self._loaded_inputs["t1"] is not None
            and self._loaded_inputs["t1"][3] == idx
        ):
            return self._loaded_inputs["t1"]

        path = self._provided_inputs["t1"][idx]
        nifti_img, affine = read_nifti(path)
        self._loaded_inputs["t1"] = (nifti_img, affine, path, idx)
        return self._loaded_inputs["t1"]

    def get_current_mesh(self):
        """Load and return the current mesh/texture pair.

        Returns
        -------
        tuple
            ``(mesh_obj, texture_path, mesh_file_path, index)`` where ``texture_path``
            may be a validated path string or ``None``.

        Raises
        ------
        ValueError
            If no mesh is available.
        """
        if self._current_inputs["mesh"] == -1:
            raise ValueError("No mesh available.")
        idx = self._current_inputs["mesh"]
        if (
            self._loaded_inputs["mesh"] is not None
            and self._loaded_inputs["mesh"][3] == idx
        ):
            return self._loaded_inputs["mesh"]

        path = self._provided_inputs["mesh"][idx]
        texture = None
        if self._current_inputs["mesh_texture"] != -1:
            texture = self._provided_inputs["mesh_texture"][
                self._current_inputs["mesh_texture"]
            ]
        mesh_obj, texture_path = read_mesh(path, texture=texture)
        self._loaded_inputs["mesh"] = (mesh_obj, texture_path, path, idx)
        return self._loaded_inputs["mesh"]

    def set_current_mesh_pair(self, index):
        """Select a mesh/texture pair by list index.

        Parameters
        ----------
        index : int
            Index into the paired mesh and mesh_texture lists.
        """
        if index < 0 or index >= len(self._provided_inputs["mesh"]):
            raise ValueError("Invalid mesh index.")
        self._current_inputs["mesh"] = index
        self._current_inputs["mesh_texture"] = index
        self._loaded_inputs["mesh"] = None

    def update_current_mesh_texture(self, texture_path):
        """Replace the texture path for the currently selected mesh pair.

        Parameters
        ----------
        texture_path : str
            Path to the new texture image.
        """
        if self._current_inputs["mesh"] == -1:
            return
        idx = self._current_inputs["mesh"]
        self._provided_inputs["mesh_texture"][idx] = texture_path
        self._loaded_inputs["mesh"] = None

    def remove_mesh_pair(self, index):
        """Remove a mesh and its texture entry at the given index.

        Parameters
        ----------
        index : int
            Index of the pair to remove.
        """
        if index < 0 or index >= len(self._provided_inputs["mesh"]):
            raise ValueError("Invalid mesh index.")
        del self._provided_inputs["mesh"][index]
        del self._provided_inputs["mesh_texture"][index]
        self._loaded_inputs["mesh"] = None

        n = len(self._provided_inputs["mesh"])
        if n == 0:
            self._current_inputs["mesh"] = -1
            self._current_inputs["mesh_texture"] = -1
        else:
            cur = self._current_inputs["mesh"]
            if cur == index:
                self._current_inputs["mesh"] = min(index, n - 1)
                self._current_inputs["mesh_texture"] = self._current_inputs["mesh"]
            elif cur > index:
                self._current_inputs["mesh"] = cur - 1
                self._current_inputs["mesh_texture"] = self._current_inputs["mesh"]

    def get_current_roi(self):
        """Load and return the current ROI.

        Returns
        -------
        tuple
            ``(roi_volume, affine, path, index)`` for the current ROI.

        Raises
        ------
        ValueError
            If no ROI is available.
        """
        if self._current_inputs["roi"] == -1:
            raise ValueError("No ROI available.")
        idx = self._current_inputs["roi"]
        if (
            self._loaded_inputs["roi"] is not None
            and self._loaded_inputs["roi"][3] == idx
        ):
            return self._loaded_inputs["roi"]

        path = self._provided_inputs["roi"][idx]
        roi, affine = read_nifti(path)
        roi = _as_filter_volume(roi)
        self._loaded_inputs["roi"] = (roi, affine, path, idx)
        return self._loaded_inputs["roi"]

    def get_roi_at(self, index):
        """Return the ROI volume and affine for a given index.

        Parameters
        ----------
        index : int
            Index into the provided ROI path list.

        Returns
        -------
        tuple
            ``(roi_volume, affine, path, index)`` for the requested ROI.

        Raises
        ------
        ValueError
            If ``index`` is out of range.
        """
        if index < 0 or index >= len(self._provided_inputs["roi"]):
            raise ValueError("Invalid ROI index.")
        path = self._provided_inputs["roi"][index]
        cached = self._loaded_rois.get(path)
        if cached is not None:
            return cached
        roi, affine = read_nifti(path)
        roi = _as_filter_volume(roi)
        loaded = (roi, affine, path, index)
        self._loaded_rois[path] = loaded
        return loaded

    def remove_roi(self, index):
        """Remove an ROI entry at the given index.

        Parameters
        ----------
        index : int
            Index of the ROI to remove.
        """
        if index < 0 or index >= len(self._provided_inputs["roi"]):
            raise ValueError("Invalid ROI index.")
        path = self._provided_inputs["roi"][index]
        del self._provided_inputs["roi"][index]
        self._loaded_rois.pop(path, None)

        n = len(self._provided_inputs["roi"])
        if n == 0:
            self._current_inputs["roi"] = -1
            self._loaded_inputs["roi"] = None
            return
        cur = self._current_inputs["roi"]
        if cur == index:
            self._current_inputs["roi"] = min(index, n - 1)
            self._loaded_inputs["roi"] = None
        elif cur > index:
            self._current_inputs["roi"] = cur - 1

    @property
    def provided_roi_paths(self):
        """Return paths or synthetic identifiers for loaded ROIs.

        Returns
        -------
        list[str]
            ROI file paths and in-memory ROI identifiers.
        """
        return list(self._provided_inputs["roi"])

    def get_current_parcel(self):
        """Load and return the current parcel data.

        Returns
        -------
        tuple
            ``(points, colors, path, index)`` for the current parcel file.

        Raises
        ------
        ValueError
            If no parcel is available.
        """
        if self._current_inputs["parcel"] == -1:
            raise ValueError("No parcel available.")
        idx = self._current_inputs["parcel"]
        if (
            self._loaded_inputs["parcel"] is not None
            and self._loaded_inputs["parcel"][3] == idx
        ):
            return self._loaded_inputs["parcel"]

        path = self._provided_inputs["parcel"][idx]
        points, colors = read_csv(path, delimiter=" ", has_header=False)
        self._loaded_inputs["parcel"] = (points, colors, path, idx)
        return self._loaded_inputs["parcel"]

    @property
    def has_input(self):
        """Check if any input is available.

        Returns
        -------
        bool
            True if at least one input type has at least one item, False
            otherwise.
        """
        return (
            self.has_tractogram
            or self.has_t1
            or self.has_mesh
            or self.has_roi
            or self.has_parcel
        )

    @property
    def has_t1(self):
        """Check if a T1 image is available.

        Returns
        -------
        bool
            True if a T1 image is available, False otherwise.
        """
        return len(self._provided_inputs["t1"]) > 0

    @property
    def has_tractogram(self):
        """Check if a tractogram is available.

        Returns
        -------
        bool
            True if a tractogram is available, False otherwise.
        """
        return len(self._provided_inputs["tractogram"]) > 0

    @property
    def has_mesh(self):
        """Check if a mesh is available.

        Returns
        -------
        bool
            True if a mesh is available, False otherwise.
        """
        return len(self._provided_inputs["mesh"]) > 0

    @property
    def has_roi(self):
        """Check if an ROI is available.

        Returns
        -------
        bool
            True if an ROI is available, False otherwise.
        """
        return len(self._provided_inputs["roi"]) > 0

    @property
    def has_parcel(self):
        """Check if a parcel is available.

        Returns
        -------
        bool
            True if a parcel is available, False otherwise.
        """
        return len(self._provided_inputs["parcel"]) > 0

    @property
    def provided_images(self):
        """Return the provided images.

        Returns
        -------
        list[str]
            List of paths to the provided images.
        """
        return self._provided_inputs["t1"]

    @property
    def provided_mesh_paths(self):
        """Return paths for loaded mesh files.

        Returns
        -------
        list[str]
            Mesh file paths paired with ``provided_mesh_texture_paths``.
        """
        return list(self._provided_inputs["mesh"])

    @property
    def provided_mesh_texture_paths(self):
        """Return texture paths paired with each mesh path.

        Returns
        -------
        list[str]
            Mesh texture file paths.
        """
        return list(self._provided_inputs["mesh_texture"])

    def get_current_mesh_pair_paths(self):
        """Return (mesh_path, texture_path) for the current selection.

        Returns
        -------
        mesh_path : str
            Currently selected mesh path.
        texture_path : str or None
            Texture path paired with the selected mesh, or None.

        Raises
        ------
        ValueError
            If no mesh is available.
        """
        if self._current_inputs["mesh"] == -1:
            raise ValueError("No mesh available.")
        idx = self._current_inputs["mesh"]
        mesh_path = self._provided_inputs["mesh"][idx]
        tex_path = None
        if idx < len(self._provided_inputs["mesh_texture"]):
            tex_path = self._provided_inputs["mesh_texture"][idx]
        return mesh_path, tex_path

    @property
    def current_mesh_index(self):
        """Index of the selected mesh/texture pair.

        Returns
        -------
        int
            Selected mesh index, or -1 if no mesh is selected.
        """
        return self._current_inputs["mesh"]

    @property
    def provided_parcel_paths(self):
        """Return paths for loaded parcel files.

        Returns
        -------
        list[str]
            Parcel file paths.
        """
        return list(self._provided_inputs["parcel"])

    @property
    def current_parcel_index(self):
        """Index of the selected parcel file.

        Returns
        -------
        int
            Selected parcel index, or -1 if no parcel is selected.
        """
        return self._current_inputs["parcel"]

    def set_current_parcel(self, index):
        """Select a parcel file by list index.

        Parameters
        ----------
        index : int
            Index into the parcel list.
        """
        if index < 0 or index >= len(self._provided_inputs["parcel"]):
            raise ValueError("Invalid parcel index.")
        self._current_inputs["parcel"] = index
        self._loaded_inputs["parcel"] = None

    def remove_parcel(self, index):
        """Remove a parcel entry at the given index.

        Parameters
        ----------
        index : int
            Index of the parcel to remove.
        """
        if index < 0 or index >= len(self._provided_inputs["parcel"]):
            raise ValueError("Invalid parcel index.")
        del self._provided_inputs["parcel"][index]
        self._loaded_inputs["parcel"] = None

        n = len(self._provided_inputs["parcel"])
        if n == 0:
            self._current_inputs["parcel"] = -1
        else:
            cur = self._current_inputs["parcel"]
            if cur == index:
                self._current_inputs["parcel"] = min(index, n - 1)
            elif cur > index:
                self._current_inputs["parcel"] = cur - 1


input_manager = InputManager()
