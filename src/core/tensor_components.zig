const std = @import("std");
const UT = @import("utils.zig");
const SC = @import("scalar.zig");

pub const SizeType = usize;
pub const IndexType = usize;
pub const Sizes = []const SizeType;
pub const Strides = []const SizeType;

pub const SliceUnion = union(SC.Tag) {
    r16: []SC.r16,
    r32: []SC.r32,
    r64: []SC.r64,

    pub fn init(slice: anytype) SliceUnion {
        return switch (std.meta.Child(@TypeOf(slice))) {
            SC.r16 => .{ .r16 = slice },
            SC.r32 => .{ .r32 = slice },
            SC.r64 => .{ .r64 = slice },
            else => @compileError("Invalid Type for SliceUnion."),
        };
    }

    pub fn len(self: SliceUnion) SizeType {
        return switch (self) {
            inline else => |x| x.len,
        };
    }

    pub fn opaque_ptr(self: anytype) *anyopaque {
        return switch (self.*) {
            inline else => |x| @ptrCast(x.ptr),
        };
    }

    pub fn bytes(self: anytype) []u8 {
        return switch(self.*) {
            .r16 => std.mem.sliceAsBytes(self.r16),
            .r32 => std.mem.sliceAsBytes(self.r32),
            .r64 => std.mem.sliceAsBytes(self.r64),
        };
    }

    pub fn dtype(self: SliceUnion) SC.Tag {
        return @as(SC.Tag, self); 
    }

    pub inline fn cast(self: SliceUnion, comptime T: type) []T {
        return @field(self, SC.name(T));
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
