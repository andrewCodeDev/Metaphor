const std = @import("std");

const FileGen = @import("file_gen.zig");
const ScriptCompiler = @import("script_compiler.zig");

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

    const exe = b.addExecutable(.{
        .name = "Cuda",
        .root_source_file = .{ .path = "src/main.zig" },
        .target = target,
        .optimize = optimize,
    });

    const gen: *FileGen = FileGen.init(.{
        .source_extension = ".cu",
        .source_directory = "src/cuda/nvcc_source",
        .target_directory = "src/cuda/nvcc_target",
        .zigsrc_directory = "src",
    });

    defer gen.deinit();

    ScriptCompiler.setupConfig(b.allocator, gen.current_directory);

    const src_body = gen.appendCudaDirectory("device_utils.cu");
    const src_head = gen.appendCudaDirectory("device_utils.h");
    const trg_lib = gen.appendLibraryDirectory("libdev_utils.so");

    if (FileGen.isModified(src_body, trg_lib) or FileGen.isModified(src_head, trg_lib))
        ScriptCompiler.compileSingleFile(b.allocator, src_body, trg_lib);

    // ensures that the @cImport always has the correct
    // absolute paths for importing C-files into Zig
    gen.makeCImport();

    gen.generate(); // try to create kernels

    exe.addLibraryPath(.{ .path = gen.appendZigsrcDirectory("lib") });
    exe.addLibraryPath(.{ .path = "dependencies/cuda/lib64" });
    exe.addLibraryPath(.{ .path = "dependencies/cuda/targets/x86_64-linux/lib/stubs" });
    exe.linkSystemLibrary("cuda");
    exe.linkSystemLibrary("cudart");
    exe.linkSystemLibrary("nvrtc");
    exe.linkSystemLibrary("dev_utils");

    exe.addObjectFile(.{
       .path = "src/lib/mp_kernels.a" 
    });
    
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
    const test_step = b.step("test-zig", "Run unit tests");
    test_step.dependOn(&run_lib_unit_tests.step);
    test_step.dependOn(&run_exe_unit_tests.step);
}
