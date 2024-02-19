const std = @import("std");
const cuda = @import("cimport.zig").C;

pub fn synchronize() void {
    cuda.mpDeviceSynchronize();
}

pub fn alloc(comptime T: type, N: usize) []T {
    var ptr: *anyopaque = undefined; 
    cuda.mpMemAlloc(@ptrCast(@alignCast(&ptr)), @sizeOf(T) * N);
    const out: [*]T = @ptrCast(@alignCast(ptr));
    return out[0..N];
}

pub fn create(comptime T: type) *T {
    var ptr: *anyopaque = undefined;
    cuda.mpMemAlloc(@ptrCast(@alignCast(&ptr)), @sizeOf(T));
    return @ptrCast(@alignCast(ptr));
}

pub fn copyToDevice(src: anytype, dst: @TypeOf(src)) void {
    switch (@typeInfo(@TypeOf(src))) {
        .Pointer => |p| {
            const T = p.child;
            if (p.size == .Slice) {
                std.debug.assert(src.len == dst.len);
                cuda.mpMemcpyHtoD(dst.ptr, src.ptr, src.len * @sizeOf(T));
            } else {
                cuda.mpMemcpyHtoD(dst, src, @sizeOf(T));
            }
        },
        else => @compileError(
            "Invalid type for CopyTo: " ++ @typeName(@TypeOf(src))
        ),
    }    
} 

pub fn copyFromDevice(src: anytype, dst: @TypeOf(src)) void {
    switch (@typeInfo(@TypeOf(src))) {
        .Pointer => |p| {
            const T = p.child;
            if (p.size == .Slice) {
                std.debug.assert(src.len == dst.len);
                cuda.mpMemcpyDtoH(dst.ptr, src.ptr, src.len * @sizeOf(T));
            } else {
                cuda.mpMemcpyDtoH(dst, src, @sizeOf(T));
            }
        },
        else => @compileError(
            "Invalid type for CopyFrom: " ++ @typeName(@TypeOf(src))
        ),
    }    
} 

pub fn free(dev_mem: anytype) void {
    switch (@typeInfo(@TypeOf(dev_mem))) {
        .Pointer => |p| {
            if (p.size == .Slice) {
                cuda.mpMemFree(dev_mem.ptr);
            } else {
                cuda.mpMemFree(dev_mem);
            }
        },
        else => @compileError(
            "Invalid type for Free: " ++ @typeName(@TypeOf(dev_mem))
        ),
    }
}

