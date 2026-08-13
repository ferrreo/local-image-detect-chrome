const std = @import("std");
const Io = std.Io;
const preprocess = @import("preprocess.zig");
const ort_session = @import("ort_session.zig");
const cstdio = @cImport({
    @cInclude("stdio.h");
});

const ModelKind = enum { distilled, forensics };

const ModelSpec = struct {
    kind: ModelKind,
    path: []const u8,
    size: usize,
    ai_label_index: u8,
    graph_opt_disabled: bool,
    mean_std: preprocess.MeanStd,
};

const Engine = struct {
    allocator: std.mem.Allocator,
    distilled: ?ort_session.Session = null,
    forensics: ?ort_session.Session = null,
    distilled_spec: ModelSpec,
    forensics_spec: ModelSpec,

    fn ensure(self: *Engine, kind: ModelKind) !*ort_session.Session {
        switch (kind) {
            .distilled => {
                if (self.distilled == null) {
                    self.distilled = try ort_session.Session.init(
                        self.allocator,
                        self.distilled_spec.path,
                        self.distilled_spec.graph_opt_disabled,
                    );
                }
                return &self.distilled.?;
            },
            .forensics => {
                if (self.forensics == null) {
                    self.forensics = try ort_session.Session.init(
                        self.allocator,
                        self.forensics_spec.path,
                        self.forensics_spec.graph_opt_disabled,
                    );
                }
                return &self.forensics.?;
            },
        }
    }

    fn deinit(self: *Engine) void {
        if (self.distilled) |*s| s.deinit();
        if (self.forensics) |*s| s.deinit();
    }
};

fn resolveRepoRoot(allocator: std.mem.Allocator, io: Io) ![]u8 {
    if (std.c.getenv("TRUEPIXEL_ROOT")) |root_z| {
        return try allocator.dupe(u8, std.mem.span(root_z));
    }
    var buf: [Io.Dir.max_path_bytes]u8 = undefined;
    const len = try std.process.currentPath(io, &buf);
    const cwd = buf[0..len];
    if (std.mem.endsWith(u8, cwd, "native/zig-infer")) {
        return try std.fs.path.resolve(allocator, &.{ cwd, "..", ".." });
    }
    return try allocator.dupe(u8, cwd);
}

fn inferOne(
    engine: *Engine,
    io: Io,
    kind: ModelKind,
    image_bytes: []const u8,
) !struct { ai: f32, ms: f64 } {
    const spec = switch (kind) {
        .distilled => engine.distilled_spec,
        .forensics => engine.forensics_spec,
    };
    const session = try engine.ensure(kind);
    const t0 = Io.Clock.awake.now(io);
    var tensor = try preprocess.tensorFromImageBytes(
        engine.allocator,
        image_bytes,
        spec.size,
        spec.mean_std,
    );
    defer tensor.deinit();
    const ai = try session.runAiProb(tensor.data, spec.size, spec.ai_label_index);
    const elapsed = t0.durationTo(Io.Clock.awake.now(io));
    const ms = @as(f64, @floatFromInt(elapsed.toNanoseconds())) / 1_000_000.0;
    return .{ .ai = ai, .ms = ms };
}

fn jsonStringAlloc(allocator: std.mem.Allocator, obj: []const u8, key: []const u8) !?[]u8 {
    var keybuf: [64]u8 = undefined;
    const needle = try std.fmt.bufPrint(&keybuf, "\"{s}\"", .{key});
    const idx = std.mem.indexOf(u8, obj, needle) orelse return null;
    var p = obj[idx + needle.len ..];
    while (p.len > 0 and (p[0] == ' ' or p[0] == '\t' or p[0] == ':')) p = p[1..];
    if (p.len == 0 or p[0] != '"') return null;
    var i: usize = 1;
    while (i < p.len) : (i += 1) {
        if (p[i] == '"' and p[i - 1] != '\\') {
            return try allocator.dupe(u8, p[1..i]);
        }
    }
    return null;
}

fn trimLine(line: []u8) []u8 {
    var s = line;
    while (s.len > 0 and (s[s.len - 1] == '\n' or s[s.len - 1] == '\r')) {
        s = s[0 .. s.len - 1];
    }
    return s;
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    const root = try resolveRepoRoot(gpa, io);
    defer gpa.free(root);

    const distilled_path = try std.fs.path.join(gpa, &.{
        root,
        "models/ai-image-detect-distilled/model_fp16.onnx",
    });
    defer gpa.free(distilled_path);
    const forensics_path = try std.fs.path.join(gpa, &.{
        root,
        "models/community-forensics/model_q4.onnx",
    });
    defer gpa.free(forensics_path);

    var engine = Engine{
        .allocator = gpa,
        .distilled_spec = .{
            .kind = .distilled,
            .path = distilled_path,
            .size = 224,
            .ai_label_index = 0,
            .graph_opt_disabled = true,
            .mean_std = preprocess.half,
        },
        .forensics_spec = .{
            .kind = .forensics,
            .path = forensics_path,
            .size = 384,
            .ai_label_index = 1,
            .graph_opt_disabled = false,
            .mean_std = preprocess.half,
        },
    };
    defer engine.deinit();

    var stdout_buf: [64 * 1024]u8 = undefined;
    var stdout_file_writer = Io.File.stdout().writer(io, &stdout_buf);
    const stdout = &stdout_file_writer.interface;

    try stdout.print(
        "{{\"event\":\"ready\",\"backend\":\"zig-ort\",\"cpu\":\"x86_64_v3\",\"eps\":[\"WebGPU\",\"Vulkan\",\"XNNPACK\",\"CPU\"]}}\n",
        .{},
    );
    try stdout.flush();

    // libc fgets: reliable line framing + immediate EOF exit (no Io.Reader hangs).
    var line_buf: [1024 * 1024]u8 = undefined;

    while (cstdio.fgets(&line_buf, @intCast(line_buf.len), cstdio.stdin)) |raw| {
        const line = trimLine(std.mem.span(raw));
        if (line.len == 0) continue;

        if (std.mem.indexOf(u8, line, "\"cmd\":\"ping\"") != null) {
            try stdout.print("{{\"ok\":true,\"event\":\"pong\"}}\n", .{});
            try stdout.flush();
            continue;
        }
        if (std.mem.indexOf(u8, line, "\"cmd\":\"quit\"") != null) {
            try stdout.print("{{\"ok\":true,\"event\":\"bye\"}}\n", .{});
            try stdout.flush();
            break;
        }
        if (std.mem.indexOf(u8, line, "\"cmd\":\"warm\"") != null) {
            const t0 = Io.Clock.awake.now(io);
            _ = try engine.ensure(.distilled);
            _ = try engine.ensure(.forensics);
            const elapsed = t0.durationTo(Io.Clock.awake.now(io));
            const ms = @as(f64, @floatFromInt(elapsed.toNanoseconds())) / 1_000_000.0;
            const ep_d = if (engine.distilled) |s| @tagName(s.ep) else "none";
            const ep_f = if (engine.forensics) |s| @tagName(s.ep) else "none";
            try stdout.print(
                "{{\"ok\":true,\"event\":\"warm\",\"ms\":{d:.3},\"distilledEp\":\"{s}\",\"forensicsEp\":\"{s}\"}}\n",
                .{ ms, ep_d, ep_f },
            );
            try stdout.flush();
            continue;
        }

        const path = try jsonStringAlloc(gpa, line, "path");
        defer if (path) |p| gpa.free(p);
        if (path == null) {
            try stdout.print("{{\"ok\":false,\"error\":\"missing path\"}}\n", .{});
            try stdout.flush();
            continue;
        }

        const run_d = if (std.mem.indexOf(u8, line, "\"models\"") == null) true else (std.mem.indexOf(u8, line, "distilled") != null);
        const run_f = if (std.mem.indexOf(u8, line, "\"models\"") == null) true else (std.mem.indexOf(u8, line, "forensics") != null);

        const bytes = Io.Dir.cwd().readFileAlloc(io, path.?, gpa, .unlimited) catch |err| {
            try stdout.print("{{\"ok\":false,\"error\":\"read failed\",\"detail\":\"{s}\"}}\n", .{@errorName(err)});
            try stdout.flush();
            continue;
        };
        defer gpa.free(bytes);

        var distilled_ai: f32 = 0.5;
        var forensics_ai: f32 = 0.5;
        var distilled_ms: f64 = 0;
        var forensics_ms: f64 = 0;

        if (run_d) {
            const r = inferOne(&engine, io, .distilled, bytes) catch |err| {
                try stdout.print("{{\"ok\":false,\"error\":\"distilled\",\"detail\":\"{s}\"}}\n", .{@errorName(err)});
                try stdout.flush();
                continue;
            };
            distilled_ai = r.ai;
            distilled_ms = r.ms;
        }
        if (run_f) {
            const r = inferOne(&engine, io, .forensics, bytes) catch |err| {
                try stdout.print("{{\"ok\":false,\"error\":\"forensics\",\"detail\":\"{s}\"}}\n", .{@errorName(err)});
                try stdout.flush();
                continue;
            };
            forensics_ai = r.ai;
            forensics_ms = r.ms;
        }

        try stdout.print(
            "{{\"ok\":true,\"distilled\":{d:.8},\"forensics\":{d:.8},\"distilledMs\":{d:.3},\"forensicsMs\":{d:.3},\"ranDistilled\":{},\"ranForensics\":{}}}\n",
            .{ distilled_ai, forensics_ai, distilled_ms, forensics_ms, run_d, run_f },
        );
        try stdout.flush();
    }
}
