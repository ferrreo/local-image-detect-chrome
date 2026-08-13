//! Browser-facing WASM API (wasm32 + simd128).
//! Spike exports SIMD-friendly preprocess helpers. Full ONNX sessions stay on
//! the native Zig+ORT host until an ORT wasm static library is linked here.

const std = @import("std");

/// ABI version for JS glue.
export fn tp_abi_version() u32 {
    return 1;
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
/// `rgb` length must be size*size*3. `out` length must be 3*size*size.
export fn tp_rgb_to_nchw_half(
    rgb: [*]const u8,
    size: usize,
    out: [*]f32,
) void {
    const plane = size * size;
    // SIMD-ish loop: process 4 pixels when possible via explicit vectors on wasm simd128.
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

/// Reports that this build is the SIMD preprocess spike (no in-module ORT).
export fn tp_has_ort_session() u32 {
    return 0;
}

comptime {
    _ = std;
}
