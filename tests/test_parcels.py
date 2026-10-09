"""Parcel loader and actor regressions using real CSV, PLY, and FURY geometry."""

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


@pytest.mark.parametrize("header", ["x,y,z", "x,y,z,value", "x,y,z,r,g,b,scalar"])
def test_all_empty_directory_retains_first_header_schema(tmp_path, header):
    (tmp_path / "a.csv").write_text("", encoding="utf-8")
    (tmp_path / "b.csv").write_text(f"{header}\n", encoding="utf-8")
    (tmp_path / "c.csv").write_text("x,y,z,value\n", encoding="utf-8")

    points, colors, probabilities = read_parcel(str(tmp_path))

    assert points.shape == (0, 3)
    assert points.dtype == np.float32
    if header.endswith("r,g,b,scalar"):
        assert colors.shape == (0, 3)
    else:
        assert colors is None
    if header.endswith(("value", "scalar")):
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
    [("1 2 3\n", None), ("1 2 3 12 34 56\n", [[12, 34, 56]])],
    ids=["positions-only", "rgb"],
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


@pytest.mark.parametrize("column_count", [2, 8])
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


CSV_LAYOUTS = [
    ("x,y,z", "", False, False),
    ("x,y,z,R,G,B,scalar", ",12,34,56,0.5", True, True),
    ("x,y,z,HEX,scalar", ",#0C2238,0.5", True, True),
    ("x,y,z,R,G,B", ",12,34,56", True, False),
    ("x,y,z,HEX", ",#0C2238", True, False),
    ("x,y,z,scalar", ",0.5", False, True),
]


@pytest.mark.parametrize("header, suffix, has_colors, has_probabilities", CSV_LAYOUTS)
@pytest.mark.parametrize("with_header", [False, True])
@pytest.mark.parametrize("coordinates", [[[1, 2, 3]], [[1, 2, 3], [4, 5, 6]]])
def test_six_csv_layouts(
    tmp_path, header, suffix, has_colors, has_probabilities, with_header, coordinates
):
    path = tmp_path / "layout.csv"
    contents = f"{header}\n" if with_header else ""
    contents += "".join(
        ",".join(map(str, point)) + suffix + "\n" for point in coordinates
    )
    path.write_text(contents, encoding="utf-8")

    points, colors, probabilities = read_parcel(str(path))

    np.testing.assert_array_equal(points, coordinates)
    assert points.dtype == np.float32
    if has_colors:
        np.testing.assert_array_equal(colors, [[12, 34, 56]] * len(coordinates))
    else:
        assert colors is None
    if has_probabilities:
        np.testing.assert_array_equal(
            probabilities, np.full(len(coordinates), 0.5, dtype=np.float32)
        )
        assert probabilities.dtype == np.float32
    else:
        assert probabilities is None


@pytest.mark.parametrize("header, suffix, has_colors, has_probabilities", CSV_LAYOUTS)
def test_six_header_only_schemas(
    tmp_path, header, suffix, has_colors, has_probabilities
):
    path = tmp_path / "header.csv"
    path.write_text(f"{header}\n", encoding="utf-8")

    points, colors, probabilities = read_parcel(str(path))

    assert points.shape == (0, 3)
    assert points.dtype == np.float32
    if has_colors:
        assert colors.shape == (0, 3)
    else:
        assert colors is None
    if has_probabilities:
        assert probabilities.shape == (0,)
        assert probabilities.dtype == np.float32
    else:
        assert probabilities is None


@pytest.mark.parametrize(
    "contents, expected_colors, expected_probability",
    [
        ("x,y,z,HEX\n1,2,3,000001\n", [[0, 0, 1]], None),
        ("1,2,3,000001\n", None, 1),
        ("1,2,3,#000001\n", [[0, 0, 1]], None),
        ("1,2,3,1e-1\n", None, 0.1),
        ("1,2,3,000001,0.5\n", [[0, 0, 1]], 0.5),
        ("x,y,z,r,g,b,a\n1,2,3,12,34,56,0.5\n", [[12, 34, 56]], 0.5),
    ],
)
def test_csv_schema_ambiguity(
    tmp_path, contents, expected_colors, expected_probability
):
    path = tmp_path / "ambiguity.csv"
    path.write_text(contents, encoding="utf-8")
    points, colors, probabilities = read_parcel(str(path))
    np.testing.assert_array_equal(points, [[1, 2, 3]])
    if expected_colors is None:
        assert colors is None
    else:
        np.testing.assert_array_equal(colors, expected_colors)
        assert colors.shape == (1, 3)
    if expected_probability is None:
        assert probabilities is None
    else:
        np.testing.assert_array_equal(
            probabilities, np.array([expected_probability], dtype=np.float32)
        )


@pytest.mark.parametrize("label", ["value", "probability", "float", "scalar/float"])
def test_four_column_probability_header_is_positional(tmp_path, label):
    path = tmp_path / "scalar.csv"
    path.write_text(f"x,y,z,{label}\n1,2,3,0.5\n", encoding="utf-8")
    _, colors, probabilities = read_parcel(str(path))
    assert colors is None
    np.testing.assert_array_equal(probabilities, [0.5])


def test_hex_csv_comments_quotes_bom_and_alignment(tmp_path):
    path = tmp_path / "hex.csv"
    path.write_text(
        "\n# full-line comment\n X , Y , Z , HEX , scalar # header comment\n"
        '1,2,3, "#0c2238" , 0.2 # note, discarded\n'
        "4,5,6,0C2238,0.5,# trailing field\n"
        "7,8,9,#000000,0.8\n",
        encoding="utf-8-sig",
    )
    points, colors, probabilities = read_parcel(str(path))
    np.testing.assert_array_equal(points, [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    np.testing.assert_array_equal(colors, [[12, 34, 56], [12, 34, 56], [0, 0, 0]])
    np.testing.assert_array_equal(
        probabilities, np.array([0.2, 0.5, 0.8], dtype=np.float32)
    )


def test_second_hash_starts_comment_after_hex(tmp_path):
    path = tmp_path / "hex.csv"
    path.write_text("1,2,3,#0C2238 # note, ignored\n", encoding="utf-8")
    points, colors, probabilities = read_parcel(str(path))
    np.testing.assert_array_equal(points, [[1, 2, 3]])
    np.testing.assert_array_equal(colors, [[12, 34, 56]])
    assert probabilities is None


@pytest.mark.parametrize(
    "contents",
    [
        "1,2,3,#ABC,0.5",
        "1,2,3,#0C2238FF,0.5",
        "1,2,3,0x0C2238,0.5",
        "1,2,3,#0C22GG,0.5",
        "1,2,3,#０C2238,0.5",
        "1,2,3,#,0.5",
        "1,2,3,#0C2238,",
        "1,2,,#0C2238,0.5",
        "1,2,3,#0C2238,0.5\n4,5,6,#0C2238",
        "1,2,3,#0C2238\n4,5,6,0.5",
        "1,2,3,0.5\n4,5,6,#0C2238",
        "1,2,3,12,34,56,78",
        "1,2,3,-1,34,56",
        "1,2,3,256,34,56",
        "1,2,3,nan,34,56",
        "1,2,3,12,inf,56",
        "bad,2,3,#0C2238,0.5",
        "1,2,3,12,34,bad",
        "1,2,3,#0C2238,bad",
    ]
    + [f"1,2,3,#0C2238,{value}" for value in ("nan", "inf", "-0.1", "1.1")],
)
def test_malformed_csv_reports_source_path(tmp_path, contents):
    path = tmp_path / "invalid.csv"
    path.write_text(contents + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Failed to load parcel") as error:
        read_parcel(str(path))
    assert str(path) in str(error.value)
    if contents.endswith(",78"):
        assert "probabilities" in str(error.value)


def test_directory_combines_rgb_and_hex_probabilities(tmp_path):
    (tmp_path / "b.csv").write_text("4,5,6,#000001,0.8\n", encoding="utf-8")
    (tmp_path / "a.csv").write_text("1,2,3,12,34,56,0.2\n", encoding="utf-8")
    points, colors, probabilities = read_parcel(str(tmp_path))
    np.testing.assert_array_equal(points, [[1, 2, 3], [4, 5, 6]])
    np.testing.assert_array_equal(colors, [[12, 34, 56], [0, 0, 1]])
    np.testing.assert_array_equal(probabilities, np.array([0.2, 0.8], dtype=np.float32))

    (tmp_path / "b.csv").write_text("4,5,6,#000001\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Inconsistent parcel schema"):
        read_parcel(str(tmp_path))


@pytest.fixture
def parcel_managers(monkeypatch):
    from tractome.mem import _visualization_manager as module
    from tractome.mem._input_manager import InputManager
    from tractome.mem._state_manager import StateManager

    inputs = object.__new__(InputManager)
    inputs.__init__()
    state = object.__new__(StateManager)
    state.__init__()
    visualization = object.__new__(module.VisualizationManager)
    visualization.__init__()
    monkeypatch.setattr(module, "input_manager", inputs)
    monkeypatch.setattr(module, "state_manager", state)
    return inputs, state, visualization


def _assert_parcel_diameters(actor, expected):
    expected = np.asarray(expected, dtype=np.float32)
    np.testing.assert_allclose(
        actor.geometry.normals.data.reshape(-1, 6, 3)[:, :, :2],
        np.broadcast_to(expected[:, None, None], (len(expected), 6, 2)),
    )
    np.testing.assert_allclose(
        actor.billboard_sizes,
        np.broadcast_to(expected[:, None], (len(expected), 2)),
    )


@pytest.mark.parametrize("with_probabilities", [True, False])
def test_recolor_retains_filter_size_visibility_and_source_data(
    tmp_path, parcel_managers, with_probabilities
):
    inputs, state, visualization = parcel_managers
    path = tmp_path / "colored.csv"
    header = "x,y,z,HEX" + (",scalar" if with_probabilities else "")
    rows = [
        f"{index},0,0,{color}" + (f",{probability}" if with_probabilities else "")
        for index, (color, probability) in enumerate(
            [("#FF0000", 0.2), ("#00FF00", 0.5), ("#0000FF", 0.8)]
        )
    ]
    path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    inputs.add_parcel(str(path))
    state.parcel_size = 40
    state.parcel_probability_threshold = 0.5
    original = visualization.visualize_parcel()[0]
    original.visible = False
    original_positions = original.geometry.positions.data.copy()
    points, colors, probabilities, _, _ = inputs.get_current_parcel()
    source_points = points.copy()
    source_colors = colors.copy()
    source_probabilities = None if probabilities is None else probabilities.copy()

    visualization.set_parcel_color((0.2, 0.4, 0.6))

    replacement = visualization.parcel_visualizations[0]
    assert replacement is not original
    assert not replacement.visible
    np.testing.assert_array_equal(
        replacement.geometry.positions.data, original_positions
    )
    expected_colors = np.broadcast_to(
        [0.2, 0.4, 0.6], replacement.geometry.colors.data.shape
    )
    np.testing.assert_allclose(replacement.geometry.colors.data, expected_colors)
    _assert_parcel_diameters(
        replacement, [0, 0.08, 0.08] if with_probabilities else [0.08] * 3
    )
    cached_points, cached_colors, cached_probabilities, _, _ = (
        inputs.get_current_parcel()
    )
    np.testing.assert_array_equal(cached_points, source_points)
    np.testing.assert_array_equal(cached_colors, source_colors)
    if with_probabilities:
        np.testing.assert_array_equal(cached_probabilities, source_probabilities)
    else:
        assert cached_probabilities is None

    visualization.set_parcel_size(100)
    _assert_parcel_diameters(
        replacement, [0, 0.2, 0.2] if with_probabilities else [0.2] * 3
    )
    visualization.set_parcel_probability_threshold(0)
    _assert_parcel_diameters(replacement, [0.2] * 3)
    np.testing.assert_allclose(replacement.geometry.colors.data, expected_colors)
    np.testing.assert_array_equal(
        replacement.geometry.positions.data, original_positions
    )
    assert not replacement.visible


@pytest.mark.parametrize("empty_file", [True, False])
def test_recolor_empty_or_absent_parcel_does_not_create_actor(
    tmp_path, parcel_managers, empty_file
):
    inputs, _, visualization = parcel_managers
    if empty_file:
        path = tmp_path / "empty.csv"
        path.write_text("x,y,z,HEX,scalar\n", encoding="utf-8")
        inputs.add_parcel(str(path))
    assert visualization.visualize_parcel() is None
    visualization.set_parcel_color((0.2, 0.4, 0.6))
    assert visualization.parcel_visualizations is None
