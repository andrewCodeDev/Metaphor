const std = @import("std");
const ChildProcess = std.ChildProcess;

pub fn buildLibraryCompileArgv(
    allocator: std.mem.Allocator,
    targets: []const [] const u8,
    libname: []const u8
) ![]const []const u8 {

    const comp_head: []const []const u8 = &.{  
        "nvcc",  "--shared", "-o", 
    };
    const comp_tail: []const []const u8 = &.{  
        "--allow-unsupported-compiler",
        "-ccbin",
        "/usr/bin/gcc",
        "--gpu-architecture=sm_89", 
        "--compiler-options", 
        "-fPIC", 
        "-I/usr/local/cuda/include", 
        "-L/usr/local/cuda/lib", 
        "-lcudart",
        "-lcuda"
    };

    var argv = try std.ArrayListUnmanaged([]const u8).initCapacity(
        allocator, comp_head.len + targets.len + comp_tail.len + 1 // libname
    );
    
    argv.appendSliceAssumeCapacity(comp_head);
    argv.appendAssumeCapacity(libname);
    argv.appendSliceAssumeCapacity(targets);
    argv.appendSliceAssumeCapacity(comp_tail);
    std.debug.assert(argv.capacity == argv.items.len);
    return argv.allocatedSlice();
}

pub fn compileSharedLibrary(input: struct {
    allocator: std.mem.Allocator, 
    targets: []const []const u8,  
    libname: []const u8
}) void {        
    blk: { 
        const argv = buildLibraryCompileArgv(input.allocator, input.targets, input.libname) 
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
