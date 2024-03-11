const std = @import("std");
const ChildProcess = std.ChildProcess;

fn joinString(allocator: std.mem.Allocator, a: []const u8, b: []const u8)  []u8 {
    return std.mem.join(allocator, "", &.{ a, b }) catch @panic("Failed to join string.");
}

const MetaphorConfig = struct {
    setup: bool = false,
    gpu_architecture: []const u8 = "",
};

var config: MetaphorConfig = .{};

pub fn setupConfig(allocator: std.mem.Allocator, cwd_path: []const u8) void {
    const config_path = std.fs.path.join(allocator, &.{ cwd_path, "config.json" })
        catch @panic("Out of Memory");
    
    const f = std.fs.cwd().openFile(config_path, .{}) 
        catch @panic("Cannot open config.");
    
        defer f.close();
    
    const f_len = f.getEndPos() 
        catch @panic("Could not get config end position.");
    const config_string = allocator.alloc(u8, f_len) 
        catch @panic("Config: Out of memory.");
    _ = f.readAll(config_string) 
        catch @panic("Could not read file config.");
    var config_json = std.json.parseFromSlice(
        std.json.Value, std.heap.page_allocator, config_string, .{}
    ) catch @panic("Failed to parse config json.");
    
    /////////////////////////////////
    // get gpu architecture /////////
    if (config_json.value.object.get("gpu-architecture")) |gpu_arch| {
        switch (gpu_arch) {
            .string => |str| {
                if (!std.mem.startsWith(u8, str, "sm_")) {
                    @panic("\n\nInvalid in format in config for gpu-achitecture. Valid format example: \"sm_89\"\n");
                }
                config.gpu_architecture = joinString(allocator, "--gpu-architecture=", str);
            }, else => @panic("\n\nInvalid in datatype in config for gpu-achitecture.\n"),
        }
    }
    config.setup = true;
}

pub fn buildSharedLibraryArgv(
    allocator: std.mem.Allocator,
    targets: []const [] const u8,
    lib_name: []const u8
) ![]const []const u8 {

    const comp_head: []const []const u8 = &.{  
        "nvcc",  "--shared", "-o", 
    };
    const comp_tail: []const []const u8 = &.{  
        "-O3",
        "--allow-unsupported-compiler",
        "-ccbin",
        "/usr/bin/gcc",
        config.gpu_architecture, 
        "--compiler-options", 
        "-fPIC", 
        "-I/usr/local/cuda/include", 
        "-L/usr/local/cuda/lib", 
        "-lcudart",
        "-lcuda"
    };

    var argv = try std.ArrayListUnmanaged([]const u8).initCapacity(
        allocator, comp_head.len + targets.len + comp_tail.len + 1 // lib_name
    );
    
    argv.appendSliceAssumeCapacity(comp_head);
    argv.appendAssumeCapacity(lib_name);
    argv.appendSliceAssumeCapacity(targets);
    argv.appendSliceAssumeCapacity(comp_tail);
    std.debug.assert(argv.capacity == argv.items.len);
    return argv.allocatedSlice();
}

pub fn compileSharedLibrary(input: struct {
    allocator: std.mem.Allocator, 
    targets: []const []const u8,  
    lib_name: []const u8
}) void {        

    if (!config.setup) {
        @panic("Config is not setup! Call ScriptCompiler.setupConfig.");
    }

    blk: { 
        const argv = buildSharedLibraryArgv(input.allocator, input.targets, input.lib_name) 
            catch @panic("concatLibraryCompileVargs: Out Of Memory");

        defer input.allocator.free(argv);
        
        const result = ChildProcess.run(.{
            .allocator = input.allocator, .argv = argv,
        }) catch |e| {
            std.log.err("Error: {}", .{e});
            @panic("ChildProcess.run failed");
        };

        switch (result.term) {
            .Exited  => |e| {
                if (e == 0) break :blk;
                std.log.err("Exit Code: {}", .{ e });
            },
            .Signal  => |s| std.log.err("Signal: {}\n",  .{s}),
            .Stopped => |s| std.log.err("Stopped: {}\n", .{s}),
            .Unknown => |u| std.log.err("Unknown: {}\n", .{u}),
        }
        std.log.err(
            "{s}\n", .{ if (result.stderr.len > 0) result.stderr else "None" }
        );
        @panic("Failed Compilation");
    }
}

pub fn objectFilesArgv(
    allocator: std.mem.Allocator,
    mod_targets: []const []const u8,
) ![]const []const u8 {

    const comp_head: []const []const u8 = &.{  
        "nvcc", "-c", 
    };
    const comp_tail: []const []const u8 = &.{  
        "-O3",
        "--allow-unsupported-compiler",
        "-ccbin",
        "/usr/bin/gcc",
        config.gpu_architecture, 
        "--compiler-options", 
        "-I/usr/local/cuda/include", 
        "-L/usr/local/cuda/lib", 
        "-lcudart",
        "-lcuda"
    };

    var argv = try std.ArrayListUnmanaged([]const u8).initCapacity(
        allocator, comp_head.len + mod_targets.len + comp_tail.len
    );
    
    argv.appendSliceAssumeCapacity(comp_head);
    argv.appendSliceAssumeCapacity(mod_targets);
    argv.appendSliceAssumeCapacity(comp_tail);
    std.debug.assert(argv.capacity == argv.items.len);
    return argv.allocatedSlice();
}

pub fn archiveFilesArgv(
    allocator: std.mem.Allocator,
    object_abspaths: []const []const u8,
    lib_name: []const u8
) ![]const []const u8 {

    var argv = try std.ArrayListUnmanaged(
        []const u8).initCapacity(allocator, 3 + object_abspaths.len
    );
    
    argv.appendAssumeCapacity("ar");
    argv.appendAssumeCapacity("-rcs");
    argv.appendAssumeCapacity(lib_name);
    argv.appendSliceAssumeCapacity(object_abspaths);
    std.debug.assert(argv.capacity == argv.items.len);
    return argv.allocatedSlice();
}

pub fn compileStaticLibrary(input: struct {
    allocator: std.mem.Allocator, 
    modded_abspaths: []const []const u8,
    object_abspaths: []const []const u8,
    current_directory: []const u8,
    target_directory: []const u8,  
    lib_name: []const u8
}) void {        

    if (!config.setup) {
        @panic("Config is not setup! Call ScriptCompiler.setupConfig.");
    }

    // move to the target directory first
    var dir = std.fs.cwd().openDir(input.target_directory, .{})
        catch @panic("Failed to open target directory.");

        defer dir.close();

    dir.setAsCwd() 
        catch @panic("Failed to move to target directory");

    blk: { 

        { // compile modified argument files
            const argv = objectFilesArgv(input.allocator, input.modded_abspaths) 
                catch @panic("Failed to create object files argv");

            defer input.allocator.free(argv);
            
            const result = ChildProcess.run(.{
                .allocator = input.allocator, .argv = argv,
            }) catch |e| {
                std.log.err("Error: {}", .{e});
                @panic("ChildProcess.run failed");
            };

            switch (result.term) {
                .Exited  => |e| {
                    if (e != 0)
                    std.log.err("Exit Code: {}", .{ e });
                },
                .Signal  => |s| std.log.err("Signal: {}\n",  .{s}),
                .Stopped => |s| std.log.err("Stopped: {}\n", .{s}),
                .Unknown => |u| std.log.err("Unknown: {}\n", .{u}),
            }

            if (result.stderr.len > 0) {
                std.log.err("{s}\n", .{ result.stderr });
            }
        }

        { // create indexed archive file for static library
            const argv = archiveFilesArgv(input.allocator, input.object_abspaths, input.lib_name)
                catch @panic("Failed to create archive argv");

            defer input.allocator.free(argv);
            
            const result = ChildProcess.run(.{
                .allocator = input.allocator, .argv = argv,
            }) catch |e| {
                std.log.err("Error: {}", .{e});
                @panic("ChildProcess.run failed");
            };

            switch (result.term) {
                .Exited  => |e| {
                    if (e == 0) break :blk;
                    std.log.err("Exit Code: {}", .{ e });
                },
                .Signal  => |s| std.log.err("Signal: {}\n",  .{s}),
                .Stopped => |s| std.log.err("Stopped: {}\n", .{s}),
                .Unknown => |u| std.log.err("Unknown: {}\n", .{u}),
            }
            std.log.err(
                "{s}\n", .{ if (result.stderr.len > 0) result.stderr else "None" }
            );

            @panic("Failed to create static library");
        }
    }

    // move back to home
    var home = std.fs.cwd().openDir(input.current_directory, .{})
        catch @panic("Failed to open home directory.");

        defer home.close();

    home.setAsCwd()
        catch @panic("Failed to move to target directory");
}

pub fn compileSingleFile(
    allocator: std.mem.Allocator,
    source_path: []const u8,
    target_path: []const u8,
) void {
    std.log.info("Creating device utilities...\n", .{});
    
    const libgen_utils_argv: []const []const u8 = &.{  
        "nvcc",
        "--shared",
        "-o", 
        target_path, 
        source_path,
        "-O3",
        config.gpu_architecture,
        "--compiler-options",
        "-fPIC", 
        "-I/usr/local/cuda/include",
        "-L/usr/local/cuda/lib",
        "-lcudart",
        "-lcuda"
    };

    const result = std.ChildProcess.run(.{
        .allocator = allocator, .argv = libgen_utils_argv
    }) catch |e| {
        std.log.err("Error: {}", .{e});
        @panic("Failed to create libdev_utils.so");
    };

    if (result.stderr.len != 0) {
        std.log.err("Error: {s}", .{result.stderr});
        @panic("Failed to create libdev_utils.so");
    }
}
