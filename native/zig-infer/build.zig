const std = @import("std");

pub fn build(b: *std.Build) void {
    const optimize = b.standardOptimizeOption(.{});
    const wasm = b.option(bool, "wasm", "Build wasm32-freestanding SIMD library") orelse false;

    const ort_root = b.option([]const u8, "ort_root", "Path to onnxruntime package (relative to zig-infer)") orelse
        "../ort/active";
    const ort_so_version = b.option([]const u8, "ort_so_version", "libonnxruntime.so.X.Y.Z version") orelse
        "1.29.0";

    if (wasm) {
        const wasm_target = b.resolveTargetQuery(.{
            .cpu_arch = .wasm32,
            .os_tag = .freestanding,
            .cpu_features_add = std.Target.wasm.featureSet(&.{ .simd128 }),
        });
        const root = b.createModule(.{
            .root_source_file = b.path("src/wasm_api.zig"),
            .target = wasm_target,
            .optimize = optimize,
        });
        const lib = b.addLibrary(.{
            .name = "truepixel_infer",
            .linkage = .dynamic,
            .root_module = root,
        });
        lib.rdynamic = true;
        b.installArtifact(lib);
        return;
    }

    var query = b.standardTargetOptionsQueryOnly(.{});
    if (query.cpu_arch == null or query.cpu_arch == .x86_64) {
        query.cpu_model = .{ .explicit = &std.Target.x86.cpu.x86_64_v3 };
    }
    const target = b.resolveTargetQuery(query);

    const root = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    root.addIncludePath(b.path("vendor"));
    root.addIncludePath(b.path(b.fmt("{s}/include", .{ort_root})));
    root.addLibraryPath(b.path(b.fmt("{s}/lib", .{ort_root})));
    // Prefer bundled libs next to the binary ($ORIGIN/../lib).
    root.addRPathSpecial("$ORIGIN/../lib");
    root.addRPath(b.path(b.fmt("{s}/lib", .{ort_root})));
    root.linkSystemLibrary("onnxruntime", .{});
    root.linkSystemLibrary("webp", .{});
    root.addCSourceFile(.{
        .file = b.path("vendor/stb_image_impl.c"),
        .flags = &.{ "-std=c99", "-O3" },
    });

    const exe = b.addExecutable(.{
        .name = "truepixel-infer",
        .root_module = root,
    });
    b.installArtifact(exe);

    // Vendor ORT (+ WebGPU plugin) into zig-out/lib (RPATH $ORIGIN/../lib).
    // setup-ort stages libonnxruntime_providers_webgpu.so into active/lib.
    const lib_files = [_][]const u8{
        b.fmt("libonnxruntime.so.{s}", .{ort_so_version}),
        "libonnxruntime.so.1",
        "libonnxruntime.so",
        "libonnxruntime_providers_shared.so",
        "libonnxruntime_providers_webgpu.so",
    };
    for (lib_files) |name| {
        const install = b.addInstallFile(
            b.path(b.fmt("{s}/lib/{s}", .{ ort_root, name })),
            b.fmt("lib/{s}", .{name}),
        );
        b.getInstallStep().dependOn(&install.step);
    }

    const run_step = b.step("run", "Run truepixel-infer");
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_cmd.addArgs(args);
    run_step.dependOn(&run_cmd.step);
}
