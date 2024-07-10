const std = @import("std");
const UT = @import("utils.zig");
const SC = @import("scalar.zig");

pub const SizeType = usize;
pub const IndexType = usize;
pub const Sizes = []const SizeType;
pub const Strides = []const SizeType;

// Tensor pointer is opaque until casted to the
// correct type by the calling kernel. These
// are not meant to be accessed by the user
// as device memory space will cause segfaults
// if access is attempted on the host side
pub const TensorPtr = union(SC.Tag) {
    r16: *anyopaque,
    r32: *anyopaque,
    r64: *anyopaque,

    // useful for freeing data
    pub fn raw(self: TensorPtr) *anyopaque {
        return switch (self) {
            inline else => |p| p,  
        };
    }
};

// Similar to a slice with opaque data and
// runtime type information.
pub const TensorData = struct {
    ptr: TensorPtr,
    len: SizeType,

    pub fn init(tag: SC.Tag, raw: *anyopaque, len: SizeType) TensorData {
        return .{
            .ptr = switch (tag) {
                .r16 => TensorPtr{ .r16 = raw }, 
                .r32 => TensorPtr{ .r32 = raw }, 
                .r64 => TensorPtr{ .r64 = raw }, 
            },
            .len = len,
        };
    }

    pub inline fn dtype(self: TensorData) SC.Tag {
        return @as(SC.Tag, self.ptr);
    }
};

pub fn compute_strides(sizes: Sizes, strides: []SizeType) void {
    std.debug.assert(sizes.len == strides.len);

    if (sizes.len == 1) {
        strides[0] = 1;
        return {};
    }

    var i: usize = (sizes.len - 1);
    var n: SizeType = 1;

    while (i > 0) : (i -= 1) {
        strides[i] = n;
        n *= sizes[i];
    }

    strides[0] = n;
}
