const std = @import("std");

// simple timer interface...
const Stopwatch = struct {
    var timer: std.time.Timer = undefined;

    pub fn start() void {
        timer = std.time.Timer.start() catch unreachable;
    }
    pub fn stop() void {
        const elapsed = timer.read();
        std.debug.print("\nTime Elapsed: {}\n", .{ elapsed });
    }
};
