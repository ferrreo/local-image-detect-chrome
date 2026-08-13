//! Browser-facing Zig WASM exports (simd128 preprocess).
//! ORT sessions live in wasm_ort_bridge.c and are linked by emcc.

const std = @import("std");

/// ABI version for JS glue. Bump when export surface changes.
export fn tp_abi_version() u32 {
    return 2;
}

/// Softmax over 2 logits → writes p0,p1 to out[0], out[1].
export fn tp_softmax2(a: f32, b: f32, out: [*]f32) void {
    const m = @max(a, b);
    const ea = @exp(a - m);
    const eb = @exp(b - m);
    const sum = ea + eb;
    out[0] = ea / sum;
    out[1] = eb / sum;
}

/// Normalize a packed RGB888 buffer into NCHW float32 with mean/std 0.5.
export fn tp_rgb_to_nchw_half(
    rgb: [*]const u8,
    size: usize,
    out: [*]f32,
) void {
    const plane = size * size;
    var i: usize = 0;
    while (i + 4 <= plane) : (i += 4) {
        inline for (0..4) |k| {
            const p = (i + k) * 3;
            const r = @as(f32, @floatFromInt(rgb[p])) / 255.0;
            const g = @as(f32, @floatFromInt(rgb[p + 1])) / 255.0;
            const b = @as(f32, @floatFromInt(rgb[p + 2])) / 255.0;
            out[i + k] = (r - 0.5) / 0.5;
            out[plane + i + k] = (g - 0.5) / 0.5;
            out[2 * plane + i + k] = (b - 0.5) / 0.5;
        }
    }
    while (i < plane) : (i += 1) {
        const p = i * 3;
        const r = @as(f32, @floatFromInt(rgb[p])) / 255.0;
        const g = @as(f32, @floatFromInt(rgb[p + 1])) / 255.0;
        const b = @as(f32, @floatFromInt(rgb[p + 2])) / 255.0;
        out[i] = (r - 0.5) / 0.5;
        out[plane + i] = (g - 0.5) / 0.5;
        out[2 * plane + i] = (b - 0.5) / 0.5;
    }
}

/// Bilinear stretch-resize RGBA → NCHW float32 (mean/std 0.5).
/// Matches native Zig host `preprocess.tensorFromImageBytes` sampling.
export fn tp_rgba_resize_nchw(
    rgba: [*]const u8,
    src_w: usize,
    src_h: usize,
    size: usize,
    out: [*]f32,
) void {
    if (src_w == 0 or src_h == 0 or size == 0) return;
    const plane = size * size;
    const inv_x: f32 = if (size <= 1) 0 else @as(f32, @floatFromInt(src_w - 1)) / @as(f32, @floatFromInt(size - 1));
    const inv_y: f32 = if (size <= 1) 0 else @as(f32, @floatFromInt(src_h - 1)) / @as(f32, @floatFromInt(size - 1));

    var y: usize = 0;
    while (y < size) : (y += 1) {
        const fy = @as(f32, @floatFromInt(y)) * inv_y;
        const y0: usize = @intFromFloat(@floor(fy));
        const y1 = @min(y0 + 1, src_h - 1);
        const wy = fy - @as(f32, @floatFromInt(y0));
        var x: usize = 0;
        while (x < size) : (x += 1) {
            const fx = @as(f32, @floatFromInt(x)) * inv_x;
            const x0: usize = @intFromFloat(@floor(fx));
            const x1 = @min(x0 + 1, src_w - 1);
            const wx = fx - @as(f32, @floatFromInt(x0));

            inline for (0..3) |ch| {
                const p00 = (y0 * src_w + x0) * 4 + ch;
                const p01 = (y0 * src_w + x1) * 4 + ch;
                const p10 = (y1 * src_w + x0) * 4 + ch;
                const p11 = (y1 * src_w + x1) * 4 + ch;
                const v00: f32 = @floatFromInt(rgba[p00]);
                const v01: f32 = @floatFromInt(rgba[p01]);
                const v10: f32 = @floatFromInt(rgba[p10]);
                const v11: f32 = @floatFromInt(rgba[p11]);
                const top = v00 + (v01 - v00) * wx;
                const bot = v10 + (v11 - v10) * wx;
                const v = (top + (bot - top) * wy) / 255.0;
                out[ch * plane + y * size + x] = (v - 0.5) / 0.5;
            }
        }
    }
}

comptime {
    _ = std;
}
