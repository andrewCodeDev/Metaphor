const std = @import("std");

const FileGen = @import("file_gen.zig");

// Although this function looks imperative, note that its job is to
// declaratively construct a build graph that will be executed by an external
// runner.
pub fn build(b: *std.Build) void {
    // Standard target options allows the person running `zig build` to choose
    // what target to build for. Here we do not override the defaults, which
    // means any target is allowed, and the default is native. Other options
    // for restricting supported target set are available.
    const target = b.standardTargetOptions(.{});

    // Standard optimization options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall. Here we do not
    // set a preferred release mode, allowing the user to decide how to optimize.
    const optimize = b.standardOptimizeOption(.{});

    //const lib = b.addSharedLibrary(.{
    //    .name = "lib",
    //    // In this case the main source file is merely a path, however, in more
    //    // complicated build scripts, this could be a generated file.
    //    .root_source_file = .{ .path = "lib.so" },
    //    .target = target,
    //    .optimize = optimize,
    //    .link_libc = true,
    //});

    //// This declares intent for the library to be installed into the standard
    //// location when the user invokes the "install" step (the default step when
    //// running `zig build`).
    //b.installArtifact(lib);

    const exe = b.addExecutable(.{
        .name = "Cuda",
        .root_source_file = .{ .path = "src/main.zig" },
        .target = target,
        .optimize = optimize,
    });

    const gen: *FileGen = FileGen.init(.{
        .source_extension = ".cu",
        .source_directory = "src/nvcc_source",
        .target_directory = "src/nvcc_target",
        .zigsrc_directory = "src",
    });

    defer gen.deinit();

    gen_utils: {
        const src = gen.appendZigsrcDirectory("device_utils.cu");
        const trg = gen.appendZigsrcDirectory("libdev_utils.so");

        if (!FileGen.isModified(src, trg))
            break :gen_utils;

        std.log.info("Creating device utilities...\n", .{});
        
        const libgen_utils_argv: []const []const u8 = &.{  
            "nvcc",
            "--shared",
            "-o", trg, src,
            "--gpu-architecture=sm_89",
             "--compiler-options",
             "-fPIC", 
            "-I/usr/local/cuda/include",
             "-L/usr/local/cuda/lib",
             "-lcudart",
             "-lcuda"
        };

        const result = std.ChildProcess.run(.{
            .allocator = b.allocator, .argv = libgen_utils_argv
        }) catch |e| {
            std.log.err("Error: {}", .{e});
            @panic("Failed to create libdev_utils.so");
        };

        if (result.stderr.len != 0) {
            std.log.err("Error: {s}", .{result.stderr});
            @panic("Failed to create libdev_utils.so");
        }
    }

    gen.generate(); // try to create kernels

    exe.addLibraryPath(.{ .path = gen.zigsrc_directory });
    exe.addLibraryPath(.{ .path = "/usr/local/cuda/lib64" });
    exe.addLibraryPath(.{ .path = "/usr/local/cuda/targets/x86_64-linux/lib/stubs" });
    exe.linkSystemLibrary("cuda");
    exe.linkSystemLibrary("cudart");
    exe.linkSystemLibrary("nvrtc");
    exe.linkSystemLibrary("dev_utils");
    exe.linkSystemLibrary("mp_kernels");
    exe.linkLibC();

    // This declares intent for the executable to be installed into the
    // standard location when the user invokes the "install" step (the default
    // step when running `zig build`).
    b.installArtifact(exe);

    // This *creates* a Run step in the build graph, to be executed when another
    // step is evaluated that depends on it. The next line below will establish
    // such a dependency.
    const run_cmd = b.addRunArtifact(exe);

    // By making the run step depend on the install step, it will be run from the
    // installation directory rather than directly from within the cache directory.
    // This is not necessary, however, if the application depends on other installed
    // files, this ensures they will be present and in the expected location.
    run_cmd.step.dependOn(b.getInstallStep());

    // This allows the user to pass arguments to the application in the build
    // command itself, like this: `zig build run -- arg1 arg2 etc`
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    // This creates a build step. It will be visible in the `zig build --help` menu,
    // and can be selected like this: `zig build run`
    // This will evaluate the `run` step rather than the default, which is "install".
    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    // Creates a step for unit testing. This only builds the test executable
    // but does not run it.
    const lib_unit_tests = b.addTest(.{
        .root_source_file = .{ .path = "src/root.zig" },
        .target = target,
        .optimize = optimize,
    });

    const run_lib_unit_tests = b.addRunArtifact(lib_unit_tests);

    const exe_unit_tests = b.addTest(.{
        .root_source_file = .{ .path = "src/main.zig" },
        .target = target,
        .optimize = optimize,
    });

    const run_exe_unit_tests = b.addRunArtifact(exe_unit_tests);

    // Similar to creating the run step earlier, this exposes a `test` step to
    // the `zig build --help` menu, providing a way for the user to request
    // running the unit tests.
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_lib_unit_tests.step);
    test_step.dependOn(&run_exe_unit_tests.step);
}
