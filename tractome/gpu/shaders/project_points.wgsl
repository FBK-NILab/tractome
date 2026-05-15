// Snap each input point to the closest position on a triangle mesh,
// accelerated by a uniform spatial grid. One thread per point.
//
// All vec3 data is stored as tightly-packed array<f32> (3 floats per element)
// to match pygfx geometry buffer layout (12-byte stride, not the 16-byte
// stride that array<vec3<f32>> would impose).

struct Uniforms {
    bbox_min_x: f32,
    bbox_min_y: f32,
    bbox_min_z: f32,
    cell_size: f32,
    grid_dim_x: u32,
    grid_dim_y: u32,
    grid_dim_z: u32,
    n_points: u32,
    threshold: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<storage, read>       in_points:    array<f32>;
@group(0) @binding(1) var<storage, read>       vertices:     array<f32>;
@group(0) @binding(2) var<storage, read>       faces:        array<u32>;
@group(0) @binding(3) var<storage, read>       cell_offsets: array<u32>;
@group(0) @binding(4) var<storage, read>       tri_indices:  array<u32>;
@group(0) @binding(5) var<uniform>             u:            Uniforms;
@group(0) @binding(6) var<storage, read_write> out_points:   array<f32>;

fn load_vec3(buf_idx: u32) -> vec3<f32> {
    let base = buf_idx * 3u;
    return vec3<f32>(in_points[base], in_points[base + 1u], in_points[base + 2u]);
}

fn load_vertex(idx: u32) -> vec3<f32> {
    let base = idx * 3u;
    return vec3<f32>(vertices[base], vertices[base + 1u], vertices[base + 2u]);
}

fn store_out(point_idx: u32, p: vec3<f32>) {
    let base = point_idx * 3u;
    out_points[base]      = p.x;
    out_points[base + 1u] = p.y;
    out_points[base + 2u] = p.z;
}

// Christer Ericson, Real-Time Collision Detection, 5.1.5.
fn closest_point_on_triangle(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> vec3<f32> {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) { return a; }

    let bp = p - b;
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) { return b; }

    let vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        let v = d1 / (d1 - d3);
        return a + v * ab;
    }

    let cp = p - c;
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) { return c; }

    let vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        let w = d2 / (d2 - d6);
        return a + w * ac;
    }

    let va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + w * (c - b);
    }

    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    return a + ab * v + ac * w;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.n_points) { return; }

    let p = load_vec3(i);
    let bbox_min = vec3<f32>(u.bbox_min_x, u.bbox_min_y, u.bbox_min_z);
    let dims = vec3<i32>(i32(u.grid_dim_x), i32(u.grid_dim_y), i32(u.grid_dim_z));
    let inv_cs = 1.0 / u.cell_size;

    let rel = (p - bbox_min) * inv_cs;
    let cell = vec3<i32>(floor(rel));

    // Search radius (in cells) covers any triangle within `threshold` of p.
    let r = i32(ceil(u.threshold * inv_cs));
    let thresh2 = u.threshold * u.threshold;

    // NaN sentinel: any point with no triangle within `threshold` is dropped
    // by writing NaN, which pygfx skips in the renderer.
    let nan = bitcast<f32>(0x7fc00000u);
    var best_d2 = thresh2;
    var best_pos = vec3<f32>(nan, nan, nan);

    for (var dz = -r; dz <= r; dz = dz + 1) {
        let cz = cell.z + dz;
        if (cz < 0 || cz >= dims.z) { continue; }
        for (var dy = -r; dy <= r; dy = dy + 1) {
            let cy = cell.y + dy;
            if (cy < 0 || cy >= dims.y) { continue; }
            for (var dx = -r; dx <= r; dx = dx + 1) {
                let cx = cell.x + dx;
                if (cx < 0 || cx >= dims.x) { continue; }

                let cell_id =
                    u32(cz) * u.grid_dim_x * u.grid_dim_y +
                    u32(cy) * u.grid_dim_x +
                    u32(cx);
                let start = cell_offsets[cell_id];
                let end   = cell_offsets[cell_id + 1u];

                for (var k = start; k < end; k = k + 1u) {
                    let tri_id = tri_indices[k];
                    let i0 = faces[tri_id * 3u];
                    let i1 = faces[tri_id * 3u + 1u];
                    let i2 = faces[tri_id * 3u + 2u];
                    let a = load_vertex(i0);
                    let b = load_vertex(i1);
                    let c = load_vertex(i2);
                    let q = closest_point_on_triangle(p, a, b, c);
                    let diff = q - p;
                    let d2 = dot(diff, diff);
                    if (d2 < best_d2) {
                        best_d2 = d2;
                        best_pos = q;
                    }
                }
            }
        }
    }

    store_out(i, best_pos);
}
