"""OBJ corner attributes and diffuse-color decoding regressions."""

import numpy as np
import pytest

from tractome.io import read_mesh
from tractome.viz import create_mesh


def _mesh(tmp_path, text):
    path = tmp_path / "mesh.obj"
    path.write_text(text, encoding="utf-8")
    return read_mesh(str(path))[0]


@pytest.mark.parametrize("scale", [1, 255])
def test_vertex_rgb_and_black(tmp_path, scale):
    mesh = _mesh(
        tmp_path,
        f"""v 0 0 0 {scale} 0 0
v 1 0 0 0 {scale} 0
v 0 1 0 0 0 {scale} 0.5 # optional alpha
v 1 1 0 0 0 0
f 1 2 3
f 2 4 3
""",
    )
    np.testing.assert_array_equal(
        mesh.vertices[mesh.faces],
        [
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            [[1, 0, 0], [1, 1, 0], [0, 1, 0]],
        ],
    )
    np.testing.assert_array_equal(
        mesh.colors[mesh.faces],
        [
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[0, 1, 0], [0, 0, 0], [0, 0, 1]],
        ],
    )
    assert mesh.colors.dtype == np.float32


def test_material_boundary_uv_seam_and_negative_indices(tmp_path):
    (tmp_path / "two colors.mtl").write_text(
        "newmtl red\nKd 1 0 0\nnewmtl blue\nKd 0 0 1\n", encoding="utf-8"
    )
    mesh = _mesh(
        tmp_path,
        """mtllib "two colors.mtl"
v 0 0 0
v 1 0 0
v 0 1 0
v 1 1 0
vt 0 0
vt 1 0
vt 0 1
vt 1 1
vt 0.5 0.5
vn 0 0 1
usemtl red
f 1/1/1 2/2/1 3/3/1
usemtl blue
f -3/-4/-1 -1/-2/-1 -2/-1/-1
""",
    )
    np.testing.assert_array_equal(
        mesh.vertices[mesh.faces],
        [
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            [[1, 0, 0], [1, 1, 0], [0, 1, 0]],
        ],
    )
    np.testing.assert_array_equal(
        mesh.colors[mesh.faces],
        [
            [[1, 0, 0]] * 3,
            [[0, 0, 1]] * 3,
        ],
    )
    np.testing.assert_array_equal(
        mesh.texcoords[mesh.faces],
        [
            [[0, 0], [1, 0], [0, 1]],
            [[1, 0], [1, 1], [0.5, 0.5]],
        ],
    )
    np.testing.assert_array_equal(mesh.normals[mesh.faces], [[[0, 0, 1]] * 3] * 2)
    assert set(mesh.faces[0]).isdisjoint(mesh.faces[1])


def test_vertex_precedence_and_partial_coloring(tmp_path, caplog):
    (tmp_path / "colors.mtl").write_text("newmtl red\nKd 1 0 0\n")
    mesh = _mesh(
        tmp_path,
        """mtllib colors.mtl missing.mtl
v 0 0 0 0 255 0
v 1 0 0
v 0 1 0
usemtl red
f 1 2 3
usemtl unknown
f 1 3 2
""",
    )
    np.testing.assert_array_equal(
        mesh.colors[mesh.faces[0]], [[0, 1, 0], [1, 0, 0], [1, 0, 0]]
    )
    np.testing.assert_allclose(
        mesh.colors[mesh.faces[1]], [[0, 1, 0], [0.7] * 3, [0.7] * 3]
    )
    np.testing.assert_array_equal(mesh.vertices.min(axis=0), [0, 0, 0])
    np.testing.assert_array_equal(mesh.vertices.max(axis=0), [1, 1, 0])
    assert caplog.text.count("contains uncolored face corners") == 1
    assert "Unable to read OBJ material library" in caplog.text


def test_uncolored_missing_library_and_xyzw(tmp_path):
    mesh = _mesh(
        tmp_path,
        """mtllib missing.mtl
v 0 0 0 1
v 1 0 0
v 0 1 0
usemtl unknown
f 1 2 3
""",
    )
    assert mesh.colors is None
    np.testing.assert_array_equal(
        mesh.vertices[mesh.faces[0]], [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    )


@pytest.mark.parametrize("rgb", ["nan 0 0", "256 0 0", "-1 0 0", "bad 0 0"])
def test_invalid_color_preserves_geometry(tmp_path, caplog, rgb):
    (tmp_path / "colors.mtl").write_text(
        "newmtl valid\nKd 0 0 1\nnewmtl invalid\nKd inf 0 0\n"
    )
    mesh = _mesh(
        tmp_path,
        f"""mtllib colors.mtl
v 0 0 0 {rgb}
v 1 0 0
v 0 1 0
usemtl valid
f 1 2 3
""",
    )
    np.testing.assert_array_equal(mesh.colors[mesh.faces[0]], [[0, 0, 1]] * 3)
    np.testing.assert_array_equal(
        mesh.vertices[mesh.faces[0]], [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    )
    assert "Ignoring invalid OBJ vertex color" in caplog.text
    assert "Ignoring invalid MTL diffuse color" in caplog.text


@pytest.mark.parametrize("token", ["0", "4", "-4", "1/0", "1/2", "1//0", "1//2"])
def test_invalid_indices_are_rejected(tmp_path, token):
    with pytest.raises(ValueError, match="Invalid OBJ face index.*line 6"):
        _mesh(tmp_path, f"v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvn 0 0 1\nf {token} 2 3\n")


def test_fan_uv_placeholders_and_incomplete_normals(tmp_path):
    mesh = _mesh(
        tmp_path,
        """v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
vt 0.5 0.5
vn 0 0 1
f 1/1/1 2//1 3 4//1
""",
    )
    np.testing.assert_array_equal(mesh.faces, [[0, 1, 2], [0, 2, 3]])
    np.testing.assert_array_equal(mesh.texcoords, [[0.5, 0.5], [0, 0], [0, 0], [0, 0]])
    assert mesh.normals is None


def test_empty_and_faceless_vertices(tmp_path):
    empty = _mesh(tmp_path, "# empty\n")
    assert empty.vertices.shape == empty.faces.shape == (0, 3)
    assert empty.colors is empty.normals is empty.texcoords is None
    mesh = _mesh(tmp_path, "v 0 0 0 1 0 0\nv 1 0 0 0 0 0\n")
    np.testing.assert_array_equal(mesh.vertices, [[0, 0, 0], [1, 0, 0]])
    np.testing.assert_array_equal(mesh.colors, [[1, 0, 0], [0, 0, 0]])
    assert mesh.faces.shape == (0, 3)


def test_later_material_definitions_replace_earlier(tmp_path):
    (tmp_path / "first.mtl").write_text("newmtl color\nKd 1 0 0\n")
    (tmp_path / "second.mtl").write_text(
        "newmtl color\nKd 0 1 0\nnewmtl color\nKd 0 0 1\n"
    )
    mesh = _mesh(
        tmp_path,
        "mtllib first.mtl second.mtl\n"
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nusemtl color\nf 1 2 3\n",
    )
    np.testing.assert_array_equal(mesh.colors[mesh.faces[0]], [[0, 0, 1]] * 3)


def test_explicit_uniform_color_on_triangle(tmp_path):
    mesh = _mesh(tmp_path, "v 0 0 0 1 0 0\nv 1 0 0 0 1 0\nv 0 1 0 0 0 1\nf 1 2 3\n")
    color = np.array([0.2, 0.4, 0.6], dtype=np.float64)
    surface = create_mesh(mesh, color=color)
    np.testing.assert_allclose(surface.geometry.colors.data, [[0.2, 0.4, 0.6]] * 3)
    np.testing.assert_array_equal(surface.geometry.positions.data, mesh.vertices)
    np.testing.assert_array_equal(surface.geometry.indices.data, mesh.faces)


@pytest.fixture
def texture_path(tmp_path):
    from PySide6.QtGui import QColor, QImage

    image = QImage(2, 2, QImage.Format_RGB888)
    image.fill(QColor("red"))
    image.setPixelColor(1, 0, QColor("blue"))
    image.setPixelColor(1, 1, QColor("blue"))
    path = tmp_path / "texture.png"
    assert image.save(str(path))
    return str(path)


def test_texture_takes_precedence_over_black_override(tmp_path, texture_path):
    mesh = _mesh(
        tmp_path,
        """v 0 0 0 1 0 0
v 1 0 0 0 1 0
v 0 1 0 0 0 1
vt 0 0
vt 1 0
vt 0 1
f 1/1 2/2 3/3
""",
    )
    surface = create_mesh(mesh, texture=texture_path, color=(0.0, 0.0, 0.0))
    assert surface.material.map is not None
    np.testing.assert_array_equal(
        surface.geometry.texcoords.data, [[0, 1], [1, 1], [0, 0]]
    )
    assert getattr(surface.geometry, "colors", None) is None


@pytest.fixture
def mesh_managers(monkeypatch):
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


@pytest.mark.parametrize("encoded", [True, False])
def test_manager_override_reset_and_texture_transitions(
    tmp_path, texture_path, mesh_managers, encoded
):
    inputs, state, visualization = mesh_managers
    vertex_lines = (
        "v 0 0 0 1 0 0\nv 1 0 0 0 1 0\nv 0 1 0 0 0 1\n"
        if encoded
        else "v 0 0 0\nv 1 0 0\nv 0 1 0\n"
    )
    _mesh(tmp_path, vertex_lines + "vt 0 0\nvt 1 0\nvt 0 1\nf 1/1 2/2 3/3\n")
    path = str(tmp_path / "mesh.obj")
    inputs.add_mesh(path, None)
    initial = visualization.visualize_mesh()[0].geometry.colors.data.copy()
    if encoded:
        np.testing.assert_array_equal(initial, inputs.get_current_mesh()[0].colors)
    assert visualization.get_mesh_color(path) is None
    state.mesh_visible = False
    state.mesh_opacity = 35

    for color in [(0.2, 0.4, 0.6), (0.0, 0.0, 0.0), None]:
        visualization.set_mesh_color(path, color)
        surface = visualization.visualize_mesh()[0]
        np.testing.assert_allclose(
            surface.geometry.colors.data, initial if color is None else [color] * 3
        )
        assert not surface.visible
        assert surface.material.opacity == pytest.approx(0.35)
        assert surface.material.alpha_mode == "blend"
        assert not surface.material.depth_write

    override = (0.2, 0.4, 0.6)
    visualization.set_mesh_color(path, override)
    other = tmp_path / "other.obj"
    other.write_text(vertex_lines + "f 1 2 3\n", encoding="utf-8")
    inputs.add_mesh(str(other), None)
    assert visualization.get_mesh_color(str(other)) is None
    other_colors = visualization.visualize_mesh()[0].geometry.colors.data
    if encoded:
        np.testing.assert_array_equal(other_colors, initial)
    else:
        assert not np.allclose(other_colors, [override] * 3)

    inputs.set_current_mesh_pair(0)
    inputs.update_current_mesh_texture(texture_path)
    textured = visualization.visualize_mesh()[0]
    assert textured.material.map is not None
    assert getattr(textured.geometry, "colors", None) is None
    assert visualization.get_mesh_color(path) == override
    inputs.update_current_mesh_texture(None)
    np.testing.assert_allclose(
        visualization.visualize_mesh()[0].geometry.colors.data, [override] * 3
    )
    visualization.reset()
    assert visualization.get_mesh_color(path) is None


@pytest.mark.parametrize("source", ["vertex", "material"])
def test_untextured_actor_preserves_file_corner_colors(tmp_path, source):
    if source == "vertex":
        mesh = _mesh(
            tmp_path,
            """v 0 0 0 1 0 0
v 1 0 0 0 1 0
v 0 1 0 0 0 1
f 1 2 3
""",
        )
        expected = [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]
    else:
        (tmp_path / "colors.mtl").write_text(
            "newmtl red\nKd 1 0 0\nnewmtl blue\nKd 0 0 1\n", encoding="utf-8"
        )
        mesh = _mesh(
            tmp_path,
            """mtllib colors.mtl
v 0 0 0
v 1 0 0
v 0 1 0
v 1 1 0
usemtl red
f 1 2 3
usemtl blue
f 2 4 3
""",
        )
        expected = [[[1, 0, 0]] * 3, [[0, 0, 1]] * 3]
    surface = create_mesh(mesh)
    np.testing.assert_array_equal(
        surface.geometry.colors.data[surface.geometry.indices.data], expected
    )
    np.testing.assert_array_equal(surface.geometry.colors.data, mesh.colors)
