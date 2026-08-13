const std = @import("std");
const c = @cImport({
    @cInclude("onnxruntime_c_api.h");
});

const OrtApiPtr = [*c]const c.OrtApi;

pub const EpKind = enum { webgpu, vulkan, xnnpack, cpu };

fn check(status: ?*c.OrtStatus, api: OrtApiPtr) !void {
    if (status) |s| {
        const msg = api.*.GetErrorMessage.?(s);
        std.log.err("ORT error: {s}", .{msg});
        api.*.ReleaseStatus.?(s);
        return error.OrtStatus;
    }
}

fn tryAppendProvider(api: OrtApiPtr, opts: ?*c.OrtSessionOptions, name: [*:0]const u8) bool {
    const status = api.*.SessionOptionsAppendExecutionProvider.?(
        opts,
        name,
        null,
        null,
        0,
    );
    if (status) |s| {
        api.*.ReleaseStatus.?(s);
        return false;
    }
    return true;
}

fn providerAvailable(api: OrtApiPtr, needle: []const u8) bool {
    // Never call Append/CreateSession for missing GPU EPs — some ORT builds hang.
    if (needle.len == 0) return true; // default CPU
    var providers: [*c][*c]u8 = undefined;
    var len: c_int = 0;
    if (api.*.GetAvailableProviders.?(&providers, &len) != null) return false;
    defer _ = api.*.ReleaseAvailableProviders.?(providers, len);
    var i: c_int = 0;
    while (i < len) : (i += 1) {
        const name = std.mem.span(providers[@intCast(i)]);
        if (std.ascii.eqlIgnoreCase(name, needle)) return true;
        // ORT reports "CPUExecutionProvider", "WebGpuExecutionProvider", …
        if (std.ascii.indexOfIgnoreCase(name, needle) != null) return true;
    }
    return false;
}

const EpAttempt = struct {
    name: [*:0]const u8,
    kind: EpKind,
    /// Substring matched against GetAvailableProviders().
    avail_key: []const u8,
};

pub const Session = struct {
    api: OrtApiPtr,
    env: *c.OrtEnv,
    session: *c.OrtSession,
    memory_info: *c.OrtMemoryInfo,
    allocator: std.mem.Allocator,
    input_name: [:0]u8,
    output_name: [:0]u8,
    ep: EpKind,

    pub fn init(
        allocator: std.mem.Allocator,
        model_path: []const u8,
        graph_opt_disabled: bool,
        prefer_ep: ?EpKind,
    ) !Session {
        const base_ptr = c.OrtGetApiBase() orelse return error.OrtApiBase;
        const api: OrtApiPtr = base_ptr.*.GetApi.?(c.ORT_API_VERSION) orelse return error.OrtApi;

        var env: ?*c.OrtEnv = null;
        try check(api.*.CreateEnv.?(c.ORT_LOGGING_LEVEL_ERROR, "truepixel-zig", &env), api);

        const path_z = try allocator.dupeZ(u8, model_path);
        defer allocator.free(path_z);

        const level: c.GraphOptimizationLevel = if (graph_opt_disabled)
            c.ORT_DISABLE_ALL
        else
            c.ORT_ENABLE_ALL;

        // Prefer GPU (WebGPU / Vulkan), then XNNPACK, then CPU (MLAS uses AVX2/AVX-512).
        // Fresh SessionOptions per attempt so a partial EP append cannot poison later tries.
        const all_attempts = [_]EpAttempt{
            .{ .name = "WebGPU", .kind = .webgpu, .avail_key = "WebGpu" },
            .{ .name = "Vulkan", .kind = .vulkan, .avail_key = "Vulkan" },
            .{ .name = "XNNPACK", .kind = .xnnpack, .avail_key = "Xnnpack" },
            .{ .name = "", .kind = .cpu, .avail_key = "" }, // default CPU EP
        };

        var ordered: [all_attempts.len]EpAttempt = undefined;
        var ordered_len: usize = 0;
        if (prefer_ep) |pref| {
            for (all_attempts) |attempt| {
                if (attempt.kind == pref) {
                    ordered[ordered_len] = attempt;
                    ordered_len += 1;
                }
            }
            for (all_attempts) |attempt| {
                if (attempt.kind != pref) {
                    ordered[ordered_len] = attempt;
                    ordered_len += 1;
                }
            }
        } else {
            for (all_attempts) |attempt| {
                ordered[ordered_len] = attempt;
                ordered_len += 1;
            }
        }

        var session: ?*c.OrtSession = null;
        var ep: EpKind = .cpu;
        var last_err: anyerror = error.OrtStatus;

        for (ordered[0..ordered_len]) |attempt| {
            var opts: ?*c.OrtSessionOptions = null;
            try check(api.*.CreateSessionOptions.?(&opts), api);
            defer api.*.ReleaseSessionOptions.?(opts);

            try check(api.*.SetSessionGraphOptimizationLevel.?(opts, level), api);
            try check(api.*.SetIntraOpNumThreads.?(opts, 2), api);

            if (attempt.name[0] != 0) {
                if (!providerAvailable(api, attempt.avail_key)) {
                    std.log.info("ORT EP {s} not in GetAvailableProviders; skip", .{attempt.name});
                    continue;
                }
                if (!tryAppendProvider(api, opts, attempt.name)) continue;
            }

            const status = api.*.CreateSession.?(env, path_z.ptr, opts, &session);
            if (status) |s| {
                const msg = api.*.GetErrorMessage.?(s);
                std.log.warn("ORT CreateSession failed for EP {s}: {s}", .{ attempt.name, msg });
                api.*.ReleaseStatus.?(s);
                last_err = error.OrtStatus;
                session = null;
                continue;
            }
            ep = attempt.kind;
            std.log.info("ORT EP: {s}", .{@tagName(ep)});
            break;
        }

        if (session == null) return last_err;

        var memory_info: ?*c.OrtMemoryInfo = null;
        try check(api.*.CreateCpuMemoryInfo.?(c.OrtArenaAllocator, c.OrtMemTypeDefault, &memory_info), api);

        var allocator_ort: ?*c.OrtAllocator = null;
        try check(api.*.GetAllocatorWithDefaultOptions.?(&allocator_ort), api);

        var in_name: [*c]u8 = null;
        try check(api.*.SessionGetInputName.?(session, 0, allocator_ort, &in_name), api);
        var out_name: [*c]u8 = null;
        try check(api.*.SessionGetOutputName.?(session, 0, allocator_ort, &out_name), api);

        const in_copy = try allocator.dupeZ(u8, std.mem.span(@as([*:0]const u8, @ptrCast(in_name))));
        const out_copy = try allocator.dupeZ(u8, std.mem.span(@as([*:0]const u8, @ptrCast(out_name))));
        _ = api.*.AllocatorFree.?(allocator_ort, in_name);
        _ = api.*.AllocatorFree.?(allocator_ort, out_name);

        return .{
            .api = api,
            .env = env.?,
            .session = session.?,
            .memory_info = memory_info.?,
            .allocator = allocator,
            .input_name = in_copy,
            .output_name = out_copy,
            .ep = ep,
        };
    }

    pub fn deinit(self: *Session) void {
        self.api.*.ReleaseSession.?(self.session);
        self.api.*.ReleaseMemoryInfo.?(self.memory_info);
        self.api.*.ReleaseEnv.?(self.env);
        self.allocator.free(self.input_name);
        self.allocator.free(self.output_name);
        self.* = undefined;
    }

    pub fn runAiProb(self: *Session, nchw: []const f32, size: usize, ai_label_index: u8) !f32 {
        const shape = [_]i64{ 1, 3, @intCast(size), @intCast(size) };
        var input_tensor: ?*c.OrtValue = null;
        try check(self.api.*.CreateTensorWithDataAsOrtValue.?(
            self.memory_info,
            @ptrCast(@constCast(nchw.ptr)),
            nchw.len * @sizeOf(f32),
            &shape,
            shape.len,
            c.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &input_tensor,
        ), self.api);
        defer self.api.*.ReleaseValue.?(input_tensor);

        const input_names = [_][*:0]const u8{self.input_name.ptr};
        const output_names = [_][*:0]const u8{self.output_name.ptr};
        var output_tensor: ?*c.OrtValue = null;

        try check(self.api.*.Run.?(
            self.session,
            null,
            &input_names,
            &input_tensor,
            1,
            &output_names,
            1,
            &output_tensor,
        ), self.api);
        defer self.api.*.ReleaseValue.?(output_tensor);

        var out_ptr: ?*anyopaque = null;
        try check(self.api.*.GetTensorMutableData.?(output_tensor, &out_ptr), self.api);
        const logits: [*]f32 = @ptrCast(@alignCast(out_ptr));
        const a = logits[0];
        const b = logits[1];
        const m = @max(a, b);
        const ea = @exp(a - m);
        const eb = @exp(b - m);
        const sum = ea + eb;
        const p0 = ea / sum;
        const p1 = eb / sum;
        return if (ai_label_index == 0) p0 else p1;
    }
};
