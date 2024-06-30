const std = @import("std");

const FileGen = @import("file_gen.zig");
const ScriptCompiler = @import("script_compiler.zig");

pub fn build(b: *std.Build) void {
    
    /////////////////////////
    // File/Kernel Generation

    const gen: *FileGen = FileGen.init(.{
        .source_extension = ".cu",
        .source_directory = b.pathJoin(&.{ "src", "cuda", "nvcc_src" }),
        .target_directory = b.pathJoin(&.{ "src", "cuda", "nvcc_trg" }),
        .zigsrc_directory = "src",
    });

    defer gen.deinit();

    ScriptCompiler.setup_config(b, gen.current_directory);

    // TODO: Cuda dependency directly exposed here. Make this configurable.
    const src_body = gen.append_kernel_directory("device_utils.cu");
    const src_head = gen.append_kernel_directory("device_utils.h");
    const trg_lib = gen.append_library_directory("libdev_utils.so");

    if (FileGen.is_modified(src_body, trg_lib) or FileGen.is_modified(src_head, trg_lib)) {

        // Device Utils:
        //
        // Device utils are fundamental operations such as:
        //
        //    Allocating to devices
        //    Freeing from devices
        //    Copying to and from devices
        //    Synchronizing devices
        //    Resetting devices
        //    Error retrieval from devices
        //
        //    Device kernels are placed in the core section
        //    and exposed through the core module.
        //

        gen.make_device_utils(src_body, trg_lib);
    }

    // Device Kernels:
    //
    //    This decides what kind of kernels to compile.
    //    Currently, this only compiles Cuda Kernels but
    //    can be expanded in future versions.
    //
    //    Device kernels are placed in the core section
    //    and exposed through the core module.
    //

    gen.make_device_kernels();

    gen.make_core_cimport();

    // Core Module:
    // 
    // The core module is used to share objects such as:
    //
    //    1. Graphs
    //    2. Kernels
    //    3. Tensors
    //
    
    const core_module = b.addModule(
        "core", .{ .root_source_file = b.path(b.pathJoin(&.{ "src", "core", "root.zig" })) }
    );

    const mp_module = b.addModule("metaphor", .{
            .root_source_file = b.path(b.pathJoin(&.{ "src", "metaphor.zig" })),
            .imports = &.{
                .{ .name = "core", .module = core_module },
            },
    });

    const target = b.standardTargetOptions(.{});

    const optimize = b.standardOptimizeOption(.{ .preferred_optimize_mode = .ReleaseFast });

    //// reusable paths for linking libraries to source files
    //// TODO: make a flag for x86-x64-linux to support different OS's
    //// TODO: Cuda dependency directly exposed here. Make this configurable.
    const cuda_stubs = b.pathJoin(&.{ "deps", "cuda", "targets", "x86_64-linux", "lib", "stubs" });
    const cuda_lib64 = b.pathJoin(&.{ "deps", "cuda", "lib64" });
    const mp_kernels = b.pathJoin(&.{ "src", "lib", "mp_kernels.a" });
    const mp_src_lib = b.pathJoin(&.{ "src", "lib" });

    // create options for each example in src/examples/
    inline for (EXAMPLE_NAMES) |EXAMPLE_NAME| {
        const examples_step = b.step("example-" ++ EXAMPLE_NAME, "Run example \"" ++ EXAMPLE_NAME ++ "\"");

        const example = b.addExecutable(.{
            .name = EXAMPLE_NAME,
            .root_source_file = b.path(b.pathJoin(&.{ "src", "examples", EXAMPLE_NAME ++ ".zig" })),
            .target = target,
            .optimize = optimize,
        });

        example.root_module.addImport("metaphor", mp_module);

        link_libraries(b, example, mp_src_lib, cuda_lib64, cuda_stubs, mp_kernels);

        const example_run = b.addRunArtifact(example);

        if (b.args) |args| {
            example_run.addArgs(args);
        }

        examples_step.dependOn(&example_run.step);
        b.default_step.dependOn(examples_step);
    }
}

const EXAMPLE_NAMES = &[_][]const u8{
    // this is for development purposes
    "subgraphs",
    "experimental",
};

fn link_libraries(
    b: *std.Build,
    step: *std.Build.Step.Compile,
    mp_src_lib: []const u8,
    cuda_lib64: []const u8,
    cuda_stubs: []const u8,
    mp_kernels: []const u8,
) void {
    step.addLibraryPath(b.path(mp_src_lib));
    step.addLibraryPath(b.path(cuda_lib64));
    step.addLibraryPath(b.path(cuda_stubs));

    step.linkSystemLibrary("cuda");
    step.linkSystemLibrary("cudart");
    step.linkSystemLibrary("nvrtc");
    step.linkSystemLibrary("dev_utils");
    step.linkSystemLibrary("cublas");

    step.addObjectFile(b.path(mp_kernels));

    step.linkLibC();
}
