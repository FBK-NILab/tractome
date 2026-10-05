"""Parcel loader regressions using real numeric CSV and ASCII PLY files."""

import numpy as np
import pytest

from tractome.io import read_parcel
from tractome.mem._visualization_manager import _set_billboard_sphere_size
from tractome.viz import create_parcels


def _write_ply(path, *, probability_property=None, values=None):
    """Write three colored vertices and a face, without mocking polyxios."""
    probability_header = (
        f"property float {probability_property}\n" if probability_property else ""
    )
    rows = ["0 0 0 12 34 56", "1 0 0 78 90 123", "2 0 0 210 45 67"]
    if probability_property:
        rows = [f"{row} {value}" for row, value in zip(rows, values)]
    path.write_text(
        "ply\n"
        "format ascii 1.0\n"
        "element vertex 3\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n" + probability_header + "element face 1\n"
        "property list uchar int vertex_indices\n"
        "end_header\n" + "\n".join(rows) + "\n3 0 1 2\n",
        encoding="utf-8",
    )
    return path


def test_comma_header_probability_csv(tmp_path):
    path = tmp_path / "probabilities.csv"
    path.write_text(
        "x,y,z,value\n0,0,0,0.2\n1,0,0,0.5\n2,0,0,0.8\n",
        encoding="utf-8",
    )

    points, colors, probabilities = read_parcel(str(path))

    np.testing.assert_array_equal(points, [[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    assert points.dtype == np.float32
    assert colors is None
    np.testing.assert_array_equal(
        probabilities, np.array([0.2, 0.5, 0.8], dtype=np.float32)
    )
    assert probabilities.dtype == np.float32
    assert probabilities.shape == (3,)


@pytest.mark.parametrize(
    "contents, expected_points, expected_colors",
    [
        (
            "0  1\t2  12 34\t56\n3\t4  5 78\t90 123\n",
            [[0, 1, 2], [3, 4, 5]],
            [[12, 34, 56], [78, 90, 123]],
        ),
        ("1\t2  3  210\t45 67\n", [[1, 2, 3]], [[210, 45, 67]]),
    ],
    ids=["repeated-spaces-and-tabs", "single-row"],
)
def test_headerless_whitespace_rgb_csv(
    tmp_path, contents, expected_points, expected_colors
):
    path = tmp_path / "colors.csv"
    path.write_text(contents, encoding="utf-8")

    points, colors, probabilities = read_parcel(str(path))

    np.testing.assert_array_equal(points, expected_points)
    assert points.dtype == np.float32
    np.testing.assert_array_equal(colors, expected_colors)
    assert probabilities is None


@pytest.mark.parametrize("property_name", ["probability", "value"])
def test_ply_keeps_vertices_rgb_and_probabilities_ignoring_faces(
    tmp_path, property_name
):
    path = _write_ply(
        tmp_path / "vertices.ply",
        probability_property=property_name,
        values=[0.2, 0.5, 0.8],
    )

    points, colors, probabilities = read_parcel(str(path))

    assert points.shape == (3, 3)
    assert points.dtype == np.float32
    np.testing.assert_array_equal(points, [[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    np.testing.assert_array_equal(colors, [[12, 34, 56], [78, 90, 123], [210, 45, 67]])
    np.testing.assert_array_equal(
        probabilities, np.array([0.2, 0.5, 0.8], dtype=np.float32)
    )
    assert probabilities.dtype == np.float32


def test_ply_rgb_does_not_imply_probability(tmp_path):
    path = _write_ply(tmp_path / "colors.PLY")

    points, colors, probabilities = read_parcel(str(path))

    assert points.shape == (3, 3)
    np.testing.assert_array_equal(colors, [[12, 34, 56], [78, 90, 123], [210, 45, 67]])
    assert probabilities is None


@pytest.mark.parametrize(
    "contents, has_probabilities",
    [("x,y,z,value\n", True), ("", False)],
    ids=["probability-header-only", "wholly-empty"],
)
def test_empty_csv_preserves_declared_schema(tmp_path, contents, has_probabilities):
    path = tmp_path / "empty.csv"
    path.write_text(contents, encoding="utf-8")

    points, colors, probabilities = read_parcel(str(path))

    assert points.shape == (0, 3)
    assert points.dtype == np.float32
    assert colors is None
    if has_probabilities:
        assert probabilities.shape == (0,)
        assert probabilities.dtype == np.float32
    else:
        assert probabilities is None


def test_directory_sorts_csv_and_preserves_probability_alignment(tmp_path):
    (tmp_path / "b.csv").write_text("x,y,z,value\n2,0,0,0.8\n", encoding="utf-8")
    (tmp_path / "a.CSV").write_text(
        "x,y,z,value\n0,0,0,0.2\n1,0,0,0.5\n", encoding="utf-8"
    )
    (tmp_path / "empty.csv").write_text("", encoding="utf-8")
    (tmp_path / "ignored.txt").write_text("not numeric", encoding="utf-8")
    (tmp_path / "nested.csv").mkdir()

    points, colors, probabilities = read_parcel(str(tmp_path))

    np.testing.assert_array_equal(points, [[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    assert points.dtype == np.float32
    assert colors is None
    np.testing.assert_array_equal(
        probabilities, np.array([0.2, 0.5, 0.8], dtype=np.float32)
    )


def test_directory_rejects_mixed_nonempty_schemas(tmp_path):
    (tmp_path / "a.csv").write_text("0 0 0 12 34 56\n", encoding="utf-8")
    (tmp_path / "b.csv").write_text("x,y,z,value\n1,0,0,0.5\n", encoding="utf-8")

    with pytest.raises(ValueError) as error:
        read_parcel(str(tmp_path))

    assert str(tmp_path) in str(error.value)


@pytest.mark.parametrize("header", ["x,y,z", "x,y,z,value", "x,y,z,r,g,b,a"])
def test_all_empty_directory_retains_first_header_schema(tmp_path, header):
    (tmp_path / "a.csv").write_text("", encoding="utf-8")
    (tmp_path / "b.csv").write_text(f"{header}\n", encoding="utf-8")
    (tmp_path / "c.csv").write_text("x,y,z,value\n", encoding="utf-8")

    points, colors, probabilities = read_parcel(str(tmp_path))

    assert points.shape == (0, 3)
    assert points.dtype == np.float32
    if header.endswith("r,g,b,a"):
        assert colors.shape == (0, 4)
    else:
        assert colors is None
    if header.endswith("value"):
        assert probabilities.shape == (0,)
        assert probabilities.dtype == np.float32
    else:
        assert probabilities is None


@pytest.mark.parametrize("probability", ["-0.1", "1.1", "nan", "inf"])
@pytest.mark.parametrize("file_format", ["csv", "ply"])
def test_invalid_probabilities_raise_path_bearing_error(
    tmp_path, probability, file_format
):
    path = tmp_path / f"invalid.{file_format}"
    if file_format == "csv":
        path.write_text(f"x,y,z,value\n0,0,0,{probability}\n", encoding="utf-8")
    else:
        _write_ply(
            path, probability_property="probability", values=[0.2, probability, 0.8]
        )

    with pytest.raises(ValueError) as error:
        read_parcel(str(path))

    assert str(path) in str(error.value)


def test_csv_header_detection_skips_comments_and_bom(tmp_path):
    path = tmp_path / "header.CSV"
    path.write_text(
        "\n# parcel probabilities\n X , Y , Z , value\n1,2,3,0.5\n",
        encoding="utf-8-sig",
    )

    points, colors, probabilities = read_parcel(str(path))

    np.testing.assert_array_equal(points, [[1, 2, 3]])
    assert colors is None
    np.testing.assert_array_equal(probabilities, np.array([0.5], dtype=np.float32))


@pytest.mark.parametrize(
    "row, expected_colors",
    [("1 2 3\n", None), ("1 2 3 12 34 56 78\n", [[12, 34, 56, 78]])],
    ids=["positions-only", "rgba"],
)
def test_optional_csv_colors_without_probability(tmp_path, row, expected_colors):
    path = tmp_path / "optional.csv"
    path.write_text(row, encoding="utf-8")

    points, colors, probabilities = read_parcel(str(path))

    np.testing.assert_array_equal(points, [[1, 2, 3]])
    assert points.dtype == np.float32
    if expected_colors is None:
        assert colors is None
    else:
        np.testing.assert_array_equal(colors, expected_colors)
    assert probabilities is None


@pytest.mark.parametrize("column_count", [2, 5, 8])
def test_unsupported_csv_column_count_reports_path_and_count(tmp_path, column_count):
    path = tmp_path / "columns.csv"
    path.write_text(" ".join(["0"] * column_count) + "\n", encoding="utf-8")

    with pytest.raises(ValueError) as error:
        read_parcel(str(path))

    assert str(path) in str(error.value)
    assert str(column_count) in str(error.value)


@pytest.mark.parametrize("coordinate", ["nan", "inf"])
def test_nonfinite_positions_are_rejected(tmp_path, coordinate):
    path = tmp_path / "positions.csv"
    path.write_text(f"{coordinate},0,0\n", encoding="utf-8")

    with pytest.raises(ValueError) as error:
        read_parcel(str(path))

    assert str(path) in str(error.value)


def test_directory_without_csv_raises_existing_error(tmp_path):
    (tmp_path / "ignored.txt").write_text("0 0 0\n", encoding="utf-8")

    with pytest.raises(ValueError) as error:
        read_parcel(str(tmp_path))

    assert str(error.value) == f"No CSV files found in directory: {tmp_path}"


def test_probability_filter_size_and_restoration():
    points = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
    probabilities = np.array([0.2, 0.5, 0.8], dtype=np.float32)
    parcel = create_parcels(points)
    normals = parcel.geometry.normals.data.reshape(3, 6, 3)
    original_z = normals[:, :, 2].copy()
    original_positions = parcel.geometry.positions.data.copy()
    original_colors = parcel.geometry.colors.data.copy()
    sizes = parcel.billboard_sizes
    for size, threshold, expected in [
        (40, 0.5, [0.0, 0.08, 0.08]),
        (100, 0.5, [0.0, 0.2, 0.2]),
        (100, 1.0, [0.0, 0.0, 0.0]),
        (100, 0.0, [0.2, 0.2, 0.2]),
    ]:
        _set_billboard_sphere_size(
            parcel, size, probabilities=probabilities, threshold=threshold
        )
        expected = np.array(expected, dtype=np.float32)
        np.testing.assert_allclose(
            normals[:, :, :2], np.broadcast_to(expected[:, None, None], (3, 6, 2))
        )
        np.testing.assert_allclose(sizes, np.broadcast_to(expected[:, None], (3, 2)))
        assert parcel.billboard_sizes is sizes
        np.testing.assert_array_equal(normals[:, :, 2], original_z)
        np.testing.assert_array_equal(
            parcel.geometry.positions.data, original_positions
        )
        np.testing.assert_array_equal(parcel.geometry.colors.data, original_colors)
