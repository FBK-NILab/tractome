"""Binary ROI export preserves positive voxels, grid, affine and contour bounds."""

import nibabel as nib
import numpy as np
import pytest

from tractome.io import read_nifti, save_roi
from tractome.viz import create_roi

AFFINE = np.array(
    [
        [0, -2, 0, 23],
        [1.5, 0, 0, -17],
        [0, 0, 3, 9],
        [0, 0, 0, 1],
    ]
)


def _fractional_mask():
    mask = np.zeros((9, 11, 13), dtype=np.float32)
    mask[2:7, 3:9, 4:11] = 0.25
    mask[3:6, 4:8, 5:10] = 1.0
    mask[4, 5, 7] = 256.0
    mask[1, 3, 4] = mask[8, 10, 12] = 0.25
    return mask


def _world_bounds(mask, affine):
    indices = np.argwhere(mask > 0)
    if not len(indices):
        return None
    # The affine includes rotation: transform each occupied voxel center.
    centers = indices @ affine[:3, :3].T + affine[:3, 3]
    return np.array([centers.min(axis=0), centers.max(axis=0)])


@pytest.mark.parametrize("extension", [".nii", ".nii.gz"])
@pytest.mark.parametrize(
    "kind", ["fractional", "binary", "empty", "labels", "negative_nan"]
)
def test_roi_roundtrip(tmp_path, extension, kind):
    source = _fractional_mask()
    if kind == "binary":
        source = (source > 0).astype(np.uint8)
    elif kind == "empty":
        source[:] = 0
    elif kind == "labels":
        source = np.where(source > 0, 256, 0).astype(np.int32)
        source[4, 5, 7] = 1024
    elif kind == "negative_nan":
        source[source == 0] = -1
        source[0, 0, 0] = np.nan
    original = source.copy()
    source_path = tmp_path / f"source{extension}"
    nib.save(nib.Nifti1Image(source, AFFINE), source_path)
    volume, affine = read_nifti(str(source_path))
    loaded_original = volume.copy()
    occupied = source > 0
    count = np.count_nonzero(occupied)
    assert count == (0 if kind == "empty" else 212)
    physical_volume = count * abs(np.linalg.det(affine[:3, :3]))
    bounds = _world_bounds(volume, affine)
    contour_bounds = (
        create_roi(volume, affine=affine).get_world_bounding_box() if count else None
    )

    for iteration in range(3):
        before = volume.copy()
        destination = tmp_path / f"saved-{iteration}{extension}"
        save_roi(str(destination), volume, affine)
        np.testing.assert_array_equal(volume, before)
        image = nib.load(destination)
        assert image.get_data_dtype() == np.dtype(np.uint8)
        volume, affine = read_nifti(str(destination))
        assert volume.shape == source.shape
        np.testing.assert_array_equal(volume, occupied.astype(np.uint8))
        np.testing.assert_allclose(affine, AFFINE, rtol=0, atol=1e-5)
        assert np.count_nonzero(volume) * abs(
            np.linalg.det(affine[:3, :3])
        ) == pytest.approx(physical_volume)
        if count:
            np.testing.assert_allclose(
                _world_bounds(volume, affine), bounds, rtol=0, atol=1e-5
            )
            np.testing.assert_allclose(
                create_roi(volume, affine=affine).get_world_bounding_box(),
                contour_bounds,
                rtol=0,
                atol=1e-5,
            )
        else:
            assert _world_bounds(volume, affine) is None

    np.testing.assert_array_equal(source, original)
    reloaded_source, _ = read_nifti(str(source_path))
    np.testing.assert_array_equal(reloaded_source, loaded_original)
