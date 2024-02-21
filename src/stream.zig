const cu = @import("cimport.zig").C;

pub const Stream = struct {
    _stream: *cu.CUstream_st,

    pub fn init() Stream {

        var stream: cu.CUstream = undefined;
        cu.cuStreamCreate(&stream, cu.CU_STREAM_DEFAULT)
            catch |err| switch (err) {
                error.NotSupported => return error.NotSupported,
                else => unreachable,
            };

        return Stream{ 
            ._stream = stream.? 
        };
    }

    pub fn deinit(self: *Stream) void {
        // Don't handle CUDA errors here
        _ = self.synchronize();
        _ = cu.cuStreamDestroy(self._stream);
        self._stream = undefined;
    }

    pub fn synchronize(self: *const Stream) void {
        cu.cuStreamSynchronize(self._stream);
    }

    pub fn format(
        self: *const Stream,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        try std.fmt.format(writer, "CuStream(device={}, stream={*})", .{ self.device, self._stream });
    }
};
