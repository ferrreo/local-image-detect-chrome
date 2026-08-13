const std = @import("std");
const c = @cImport({
    @cInclude("stb_image.h");
    @cInclude("webp/decode.h");
});

pub const Tensor = struct {
    data: []f32,
    size: usize,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *Tensor) void {
        self.allocator.free(self.data);
        self.* = undefined;
    }
};

pub const MeanStd = struct {
    mean: [3]f32,
    std: [3]f32,
};

pub const half: MeanStd = .{
    .mean = .{ 0.5, 0.5, 0.5 },
    .std = .{ 0.5, 0.5, 0.5 },
};

const RgbImage = struct {
    pixels: [*]u8,
    w: usize,
    h: usize,
    free_fn: *const fn (?*anyopaque) callconv(.c) void,
    free_ctx: ?*anyopaque,

    fn deinit(self: RgbImage) void {
        self.free_fn(self.free_ctx);
    }
};

fn stbiFree(ctx: ?*anyopaque) callconv(.c) void {
    if (ctx) |p| c.stbi_image_free(p);
}

fn webpFree(ctx: ?*anyopaque) callconv(.c) void {
    if (ctx) |p| c.WebPFree(p);
}

fn isWebp(bytes: []const u8) bool {
    return bytes.len >= 12 and
        bytes[0] == 'R' and bytes[1] == 'I' and bytes[2] == 'F' and bytes[3] == 'F' and
        bytes[8] == 'W' and bytes[9] == 'E' and bytes[10] == 'B' and bytes[11] == 'P';
}

fn decodeRgb(bytes: []const u8) !RgbImage {
    if (isWebp(bytes)) {
        var w: c_int = 0;
        var h: c_int = 0;
        const img = c.WebPDecodeRGB(bytes.ptr, bytes.len, &w, &h);
        if (img == null) return error.ImageDecodeFailed;
        return .{
            .pixels = img,
            .w = @intCast(w),
            .h = @intCast(h),
            .free_fn = webpFree,
            .free_ctx = img,
        };
    }

    var w: c_int = 0;
    var h: c_int = 0;
    var comp: c_int = 0;
    const img = c.stbi_load_from_memory(
        bytes.ptr,
        @intCast(bytes.len),
        &w,
        &h,
        &comp,
        3,
    );
    if (img == null) return error.ImageDecodeFailed;
    return .{
        .pixels = img,
        .w = @intCast(w),
        .h = @intCast(h),
        .free_fn = stbiFree,
        .free_ctx = img,
    };
}

/// Decode image bytes, stretch-resize to size×size, emit NCHW float32 normalized tensor.
pub fn tensorFromImageBytes(
    allocator: std.mem.Allocator,
    bytes: []const u8,
    size: usize,
    ms: MeanStd,
) !Tensor {
    const img = try decodeRgb(bytes);
    defer img.deinit();

    const src_w = img.w;
    const src_h = img.h;
    const plane = size * size;
    const out = try allocator.alloc(f32, 3 * plane);
    errdefer allocator.free(out);

    const inv_x = if (size <= 1) 0 else @as(f32, @floatFromInt(src_w - 1)) / @as(f32, @floatFromInt(size - 1));
    const inv_y = if (size <= 1) 0 else @as(f32, @floatFromInt(src_h - 1)) / @as(f32, @floatFromInt(size - 1));

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
                const idx00 = (y0 * src_w + x0) * 3 + ch;
                const idx01 = (y0 * src_w + x1) * 3 + ch;
                const idx10 = (y1 * src_w + x0) * 3 + ch;
                const idx11 = (y1 * src_w + x1) * 3 + ch;
                const v00: f32 = @floatFromInt(img.pixels[idx00]);
                const v01: f32 = @floatFromInt(img.pixels[idx01]);
                const v10: f32 = @floatFromInt(img.pixels[idx10]);
                const v11: f32 = @floatFromInt(img.pixels[idx11]);
                const top = v00 + (v01 - v00) * wx;
                const bot = v10 + (v11 - v10) * wx;
                const v = (top + (bot - top) * wy) / 255.0;
                out[ch * plane + y * size + x] = (v - ms.mean[ch]) / ms.std[ch];
            }
        }
    }

    return .{ .data = out, .size = size, .allocator = allocator };
}
