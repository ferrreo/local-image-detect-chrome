const std = @import("std");
const c = @cImport({
    @cInclude("onnxruntime_c_api.h");
});

const OrtApiPtr = [*c]const c.OrtApi;

pub const EpKind = enum { cuda, webgpu, xnnpack, cpu };

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

fn tryAppendCuda(api: OrtApiPtr, opts: ?*c.OrtSessionOptions) bool {
    var cuda_opts: ?*c.OrtCUDAProviderOptionsV2 = null;
    const create_status = api.*.CreateCUDAProviderOptions.?(&cuda_opts);
    if (create_status) |s| {
        api.*.ReleaseStatus.?(s);
        return false;
    }
    defer api.*.ReleaseCUDAProviderOptions.?(cuda_opts);

    const status = api.*.SessionOptionsAppendExecutionProvider_CUDA_V2.?(opts, cuda_opts);
    if (status) |s| {
        api.*.ReleaseStatus.?(s);
        return false;
    }
    return true;
}

fn providerAvailable(api: OrtApiPtr, needle: []const u8) bool {
    if (needle.len == 0) return true;
    var providers: [*c][*c]u8 = undefined;
    var len: c_int = 0;
    if (api.*.GetAvailableProviders.?(&providers, &len) != null) return false;
    defer _ = api.*.ReleaseAvailableProviders.?(providers, len);
    var i: c_int = 0;
    while (i < len) : (i += 1) {
        const name = std.mem.span(providers[@intCast(i)]);
        if (std.ascii.indexOfIgnoreCase(name, needle) != null) return true;
    }
    return false;
}

const libc = @cImport({
    @cInclude("unistd.h");
});

fn pathExistsZ(path_z: [*:0]const u8) bool {
    return libc.access(path_z, libc.F_OK) == 0;
}

fn pathExists(allocator: std.mem.Allocator, path: []const u8) bool {
    const z = allocator.dupeZ(u8, path) catch return false;
    defer allocator.free(z);
    return pathExistsZ(z.ptr);
}

fn selfExeDir(allocator: std.mem.Allocator) ?[]const u8 {
    var buf: [Io.Dir.max_path_bytes]u8 = undefined;
    const n = libc.readlink("/proc/self/exe", &buf, buf.len);
    if (n <= 0) return null;
    const exe = buf[0..@intCast(n)];
    const dir = std.fs.path.dirname(exe) orelse return null;
    return allocator.dupe(u8, dir) catch null;
}

// Zig 0.16: std.Io is the filesystem surface; keep a light alias for max_path_bytes.
const Io = std.Io;

fn candidateWebGpuPluginPaths(allocator: std.mem.Allocator) ![][]const u8 {
    var list: std.ArrayListUnmanaged([]const u8) = .empty;
    errdefer {
        for (list.items) |p| allocator.free(p);
        list.deinit(allocator);
    }

    if (std.c.getenv("TRUEPIXEL_WEBGPU_EP")) |env_z| {
        try list.append(allocator, try allocator.dupe(u8, std.mem.span(env_z)));
    }

    if (selfExeDir(allocator)) |dir| {
        defer allocator.free(dir);
        try list.append(allocator, try std.fs.path.join(allocator, &.{ dir, "..", "lib", "libonnxruntime_providers_webgpu.so" }));
        try list.append(allocator, try std.fs.path.join(allocator, &.{ dir, "libonnxruntime_providers_webgpu.so" }));
    }

    if (std.c.getenv("TRUEPIXEL_ROOT")) |root_z| {
        const root = std.mem.span(root_z);
        try list.append(allocator, try std.fs.path.join(allocator, &.{ root, "native/ort/active/lib/libonnxruntime_providers_webgpu.so" }));
        try list.append(allocator, try std.fs.path.join(allocator, &.{ root, "native/ort/webgpu-ep/libonnxruntime_providers_webgpu.so" }));
    }

    // Relative to common cwd layouts (repo root or native/zig-infer).
    try list.append(allocator, try allocator.dupe(u8, "native/ort/active/lib/libonnxruntime_providers_webgpu.so"));
    try list.append(allocator, try allocator.dupe(u8, "native/ort/webgpu-ep/libonnxruntime_providers_webgpu.so"));
    try list.append(allocator, try allocator.dupe(u8, "../ort/active/lib/libonnxruntime_providers_webgpu.so"));
    try list.append(allocator, try allocator.dupe(u8, "../ort/webgpu-ep/libonnxruntime_providers_webgpu.so"));

    return try list.toOwnedSlice(allocator);
}

fn findWebGpuPlugin(allocator: std.mem.Allocator) ?[:0]const u8 {
    const paths = candidateWebGpuPluginPaths(allocator) catch return null;
    defer {
        for (paths) |p| allocator.free(p);
        allocator.free(paths);
    }
    for (paths) |p| {
        if (!pathExists(allocator, p)) continue;
        return allocator.dupeZ(u8, p) catch null;
    }
    return null;
}

fn registerWebGpuPlugin(api: OrtApiPtr, env: ?*c.OrtEnv, plugin_path: [:0]const u8) bool {
    const reg = api.*.RegisterExecutionProviderLibrary orelse {
        std.log.warn("ORT RegisterExecutionProviderLibrary missing from API", .{});
        return false;
    };
    const status = reg(env, "truepixel_webgpu_ep", plugin_path.ptr);
    if (status) |s| {
        const msg = api.*.GetErrorMessage.?(s);
        // Dual sessions each CreateEnv; ORT treats the registration name as once-per-process.
        if (std.ascii.indexOfIgnoreCase(std.mem.span(msg), "already registered") != null) {
            api.*.ReleaseStatus.?(s);
            return true;
        }
        std.log.warn("ORT WebGPU plugin register failed ({s}): {s}", .{ plugin_path, msg });
        api.*.ReleaseStatus.?(s);
        return false;
    }
    std.log.info("ORT WebGPU plugin registered: {s}", .{plugin_path});
    return true;
}

fn findWebGpuDevice(api: OrtApiPtr, env: ?*c.OrtEnv) ?*const c.OrtEpDevice {
    const get_devices = api.*.GetEpDevices orelse return null;
    // C API: const OrtEpDevice* const** — many-pointer of opaque values is illegal;
    // use a pointer-to-pointer table instead.
    var devices: ?[*]const *const c.OrtEpDevice = null;
    var num: usize = 0;
    if (get_devices(env, @ptrCast(&devices), &num) != null) return null;
    const table = devices orelse return null;
    const ep_name_fn = api.*.EpDevice_EpName orelse return null;
    var i: usize = 0;
    while (i < num) : (i += 1) {
        const dev = table[i];
        const name = std.mem.span(ep_name_fn(dev));
        if (std.ascii.eqlIgnoreCase(name, "WebGpuExecutionProvider") or
            std.ascii.indexOfIgnoreCase(name, "WebGpu") != null)
        {
            return dev;
        }
    }
    return null;
}

fn tryAppendWebGpu(api: OrtApiPtr, env: ?*c.OrtEnv, opts: ?*c.OrtSessionOptions) bool {
    const device = findWebGpuDevice(api, env) orelse {
        std.log.info("ORT WebGPU: no WebGpuExecutionProvider device after plugin register", .{});
        return false;
    };
    const append = api.*.SessionOptionsAppendExecutionProvider_V2 orelse return false;

    const keys = [_][*:0]const u8{"preferredLayout"};
    const vals = [_][*:0]const u8{"NCHW"};
    const devices = [_]*const c.OrtEpDevice{device};

    const status = append(
        opts,
        env,
        @ptrCast(&devices),
        1,
        &keys,
        &vals,
        1,
    );
    if (status) |s| {
        const msg = api.*.GetErrorMessage.?(s);
        std.log.warn("ORT AppendExecutionProvider_V2 WebGPU failed: {s}", .{msg});
        api.*.ReleaseStatus.?(s);
        return false;
    }
    return true;
}

const EpAttempt = struct {
    name: [*:0]const u8,
    kind: EpKind,
    /// Substring match against GetAvailableProviders. Empty = always try (CPU).
    /// "plugin" = WebGPU plugin EP path (not listed until registered).
    avail_key: []const u8,
};

/// One OrtEnv + WebGPU plugin registration per process (dual models share it).
const SharedOrt = struct {
    api: OrtApiPtr,
    env: *c.OrtEnv,
    webgpu_ok: bool,
};

var shared_ort: ?SharedOrt = null;

fn ensureSharedOrt(allocator: std.mem.Allocator) !SharedOrt {
    if (shared_ort) |s| return s;

    const base_ptr = c.OrtGetApiBase() orelse return error.OrtApiBase;
    const api: OrtApiPtr = base_ptr.*.GetApi.?(c.ORT_API_VERSION) orelse return error.OrtApi;

    var env: ?*c.OrtEnv = null;
    try check(api.*.CreateEnv.?(c.ORT_LOGGING_LEVEL_ERROR, "truepixel-zig", &env), api);

    var webgpu_ok = false;
    if (findWebGpuPlugin(allocator)) |plugin_path| {
        defer allocator.free(plugin_path);
        webgpu_ok = registerWebGpuPlugin(api, env, plugin_path);
    } else {
        std.log.info("ORT WebGPU plugin .so not found; WebGPU EP unavailable", .{});
    }

    const s: SharedOrt = .{
        .api = api,
        .env = env.?,
        .webgpu_ok = webgpu_ok,
    };
    shared_ort = s;
    return s;
}

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
        const shared = try ensureSharedOrt(allocator);
        const api = shared.api;
        const env: ?*c.OrtEnv = shared.env;
        const webgpu_ok = shared.webgpu_ok;

        const path_z = try allocator.dupeZ(u8, model_path);
        defer allocator.free(path_z);

        const level: c.GraphOptimizationLevel = if (graph_opt_disabled)
            c.ORT_DISABLE_ALL
        else
            c.ORT_ENABLE_ALL;

        // Cross-vendor native GPU = WebGPU plugin (Dawn→Vulkan on Linux).
        // CUDA only when the ORT gpu package is linked. No separate Vulkan EP.
        const all_attempts = [_]EpAttempt{
            .{ .name = "WebGPU", .kind = .webgpu, .avail_key = "plugin" },
            .{ .name = "CUDA", .kind = .cuda, .avail_key = "CUDA" },
            .{ .name = "XNNPACK", .kind = .xnnpack, .avail_key = "Xnnpack" },
            .{ .name = "", .kind = .cpu, .avail_key = "" },
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
                if (attempt.kind == .webgpu) {
                    if (!webgpu_ok) continue;
                    if (!tryAppendWebGpu(api, env, opts)) continue;
                } else {
                    if (!providerAvailable(api, attempt.avail_key)) {
                        std.log.info("ORT EP {s} not in GetAvailableProviders; skip", .{attempt.name});
                        continue;
                    }
                    const ok = if (attempt.kind == .cuda)
                        tryAppendCuda(api, opts)
                    else
                        tryAppendProvider(api, opts, attempt.name);
                    if (!ok) {
                        std.log.warn("ORT AppendExecutionProvider failed for {s}", .{attempt.name});
                        continue;
                    }
                }
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
        // OrtEnv + WebGPU plugin stay process-global (see ensureSharedOrt).
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
