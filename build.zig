const std = @import("std");

const FileGen = @import("file_gen.zig");
const ScriptCompiler = @import("script_compiler.zig");

pub fn build(b: *std.Build) void {

    /////////////////////////
    // File/Kernel Generation

    const gen: *FileGen = FileGen.init(.{
        .source_extension = ".cu",
        .source_directory = b.pathJoin(&.{"src", "cuda", "nvcc_src"}),
        .target_directory = b.pathJoin(&.{"src", "cuda", "nvcc_trg"}),
        .zigsrc_directory = "src",
    });

    defer gen.deinit();

    ScriptCompiler.setupConfig(b, gen.current_directory);

    const src_body = gen.appendCudaDirectory("device_utils.cu");
    const src_head = gen.appendCudaDirectory("device_utils.h");
    const trg_lib = gen.appendLibraryDirectory("libdev_utils.so");

    if (FileGen.isModified(src_body, trg_lib) or FileGen.isModified(src_head, trg_lib))
        ScriptCompiler.compileSingleFile(b, src_body, trg_lib);

    // ensures that the @cImport always has the correct
    // absolute paths for importing C-files into Zig
    gen.makeCImport();

    // try to create kernels
    gen.generate(); 

    // Standard target options allows the person running `zig build` to choose
    // what target to build for. Here we do not override the defaults, which
    // means any target is allowed, and the default is native. Other options
    // for restricting supported target set are available.
    const target = b.standardTargetOptions(.{});

    // Standard optimization options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall. Here we do not
    // set a preferred release mode, allowing the user to decide how to optimize.
    const optimize = b.standardOptimizeOption(.{
        .preferred_optimize_mode = .ReleaseFast
    });

    const mp_module = b.addModule("metaphor", .{
        .root_source_file = .{ .path = b.pathJoin(&.{ "src", "metaphor.zig" }) }
    });

    // reusable paths for linking libraries to source files
    // TODO: make a flag for x86-x64-linux to support different OS's
    const cuda_stubs = b.pathJoin(&.{ "deps", "cuda", "targets", "x86_64-linux", "lib", "stubs" });
    const cuda_lib64 = b.pathJoin(&.{ "deps", "cuda", "lib64" });
    const mp_kernels = b.pathJoin(&.{ "src", "lib", "mp_kernels.a" });
    const mp_src_lib = b.pathJoin(&.{ "src", "lib" });

    // create options for each example in src/examples/
    inline for (EXAMPLE_NAMES) |EXAMPLE_NAME| {

        const examples_step = b.step("example-" ++ EXAMPLE_NAME, "Run example \"" ++ EXAMPLE_NAME ++ "\"");
        
        const example = b.addExecutable(.{
            .name = EXAMPLE_NAME,
            .root_source_file = .{ .path = b.pathJoin(&.{"src", "examples", EXAMPLE_NAME ++ ".zig"}) },
            .target = target,
            .optimize = optimize,
        });

        example.root_module.addImport("metaphor", mp_module);

        linkLibraries(example, mp_src_lib, cuda_lib64, cuda_stubs, mp_kernels);

        const example_run = b.addRunArtifact(example);
        // This allows the user to pass arguments to the application in the build
        // command itself, like this: `zig build run -- arg1 arg2 etc`
        if (b.args) |args| {
            example_run.addArgs(args);
        }
        examples_step.dependOn(&example_run.step);
        b.default_step.dependOn(examples_step);
    }
    
    // Creates a step for unit testing. This only builds the test executable
    // but does not run it.
    //const lib_unit_tests = b.addTest(.{
    //    .root_source_file = .{ .path = "src/root.zig" },
    //    .target = target,
    //    .optimize = optimize,
    //});

    //const run_lib_unit_tests = b.addRunArtifact(lib_unit_tests);

    //const exe_unit_tests = b.addTest(.{
    //    .root_source_file = .{ .path = "src/main.zig" },
    //    .target = target,
    //    .optimize = optimize,
    //});

    //const run_exe_unit_tests = b.addRunArtifact(exe_unit_tests);

    // Similar to creating the run step earlier, this exposes a `test` step to
    // the `zig build --help` menu, providing a way for the user to request
    // running the unit tests.
    //const test_step = b.step("test-zig", "Run unit tests");
    //test_step.dependOn(&run_lib_unit_tests.step);
    //test_step.dependOn(&run_exe_unit_tests.step);
}

const EXAMPLE_NAMES = &[_][]const u8{
    "scalar",
    "linear_maps",
    "streams",
    "subgraphs",
    "feedforward",

    // this is for development purposes
    "experimental",
};

fn linkLibraries(
    step: *std.Build.Step.Compile, 
    mp_src_lib: []const u8,
    cuda_lib64: []const u8,
    cuda_stubs: []const u8,
    mp_kernels: []const u8
) void {    
    step.addLibraryPath(.{ .path = mp_src_lib });
    step.addLibraryPath(.{ .path = cuda_lib64 });
    step.addLibraryPath(.{ .path = cuda_stubs });

    step.linkSystemLibrary("cuda");
    step.linkSystemLibrary("cudart");
    step.linkSystemLibrary("nvrtc");
    step.linkSystemLibrary("dev_utils");

    step.addObjectFile(.{ .path = mp_kernels });

    step.linkLibC();
}
