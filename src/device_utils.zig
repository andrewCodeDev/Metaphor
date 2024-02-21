const std = @import("std");
const cuda = @import("cimport.zig").C;

pub const Stream = cuda.Stream;

pub fn initDevice(device_number: u32) void {
    cuda.initDevice(device_number);
}

pub fn initStream() Stream {
    return cuda.mpInitStream();
}

pub fn deinitStream(stream: Stream) void {
    // std.debug.assert(stream != null);
    return cuda.mpDeinitStream(stream);
}

pub fn synchronizeStream(stream: Stream) void {
    cuda.mpDeviceSynchronize(stream);
}

pub fn synchronizeDevice() void {
    cuda.mpDeviceSynchronize();
}

pub fn alloc(comptime T: type, N: usize, stream: Stream) []T {
    // std.debug.assert(stream != null);
    const ptr: *anyopaque = cuda.mpMemAlloc(@sizeOf(T) * N, stream) orelse unreachable;
    const out: [*]T = @ptrCast(@alignCast(ptr));
    return out[0..N];
}

pub fn create(comptime T: type, stream: Stream) *T {
    std.debug.assert(stream != null);
    const ptr: *anyopaque = cuda.mpMemAlloc(@sizeOf(T), stream) orelse unreachable;
    return @ptrCast(@alignCast(ptr));
}

pub fn copyToDevice(src: anytype, dst: @TypeOf(src), stream: Stream) void {
    // std.debug.assert(stream != null);
    switch (@typeInfo(@TypeOf(src))) {
        .Pointer => |p| {
            const T = p.child;
            if (p.size == .Slice) {
                std.debug.assert(src.len == dst.len);
                cuda.mpMemcpyHtoD(dst.ptr, src.ptr, src.len * @sizeOf(T), stream);
            } else {
                cuda.mpMemcpyHtoD(dst, src, @sizeOf(T), stream);
            }
        },
        else => @compileError(
            "Invalid type for CopyTo: " ++ @typeName(@TypeOf(src))
        ),
    }    
} 

pub fn copyFromDevice(src: anytype, dst: @TypeOf(src), stream: Stream) void {
    // std.debug.assert(stream != null);
    switch (@typeInfo(@TypeOf(src))) {
        .Pointer => |p| {
            const T = p.child;
            if (p.size == .Slice) {
                std.debug.assert(src.len == dst.len);
                cuda.mpMemcpyDtoH(dst.ptr, src.ptr, src.len * @sizeOf(T), stream);
            } else {
                cuda.mpMemcpyDtoH(dst, src, @sizeOf(T), stream);
            }
        },
        else => @compileError(
            "Invalid type for CopyFrom: " ++ @typeName(@TypeOf(src))
        ),
    }    
} 

pub fn free(dev_mem: anytype, stream: Stream) void {
    // std.debug.assert(stream != null);
    switch (@typeInfo(@TypeOf(dev_mem))) {
        .Pointer => |p| {
            if (p.size == .Slice) {
                cuda.mpMemFree(dev_mem.ptr, stream);
            } else {
                cuda.mpMemFree(dev_mem, stream);
            }
        },
        else => @compileError(
            "Invalid type for Free: " ++ @typeName(@TypeOf(dev_mem))
        ),
    }
}

