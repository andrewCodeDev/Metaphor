const std = @import("std");
const UT = @import("utility.zig");
const SCL = @import("scalar.zig");
const C = @import("cimport.zig").C;

pub const SizeType = usize;
pub const IndexType = usize;
pub const Sizes = []const SizeType;
pub const Strides = []const SizeType;

pub inline fn DeviceTensor(comptime DataType: type) type {
    return switch (DataType) {
        SCL.q8 => C.QTensor8,
        SCL.r16 => C.RTensor16,
        SCL.r32 => C.RTensor32,
        SCL.r64 => C.RTensor64,
        SCL.c16 => C.CTensor16,
        SCL.c32 => C.CTensor32,
        SCL.c64 => C.CTensor64,
        else => @compileError("Invalid Type for SliceUnion."),
    };
}

pub const SliceUnion = union(enum) {
    q8: []SCL.q8,
    r16: []SCL.r16,
    r32: []SCL.r32,
    r64: []SCL.r64,
    c16: []SCL.c16,
    c32: []SCL.c32,
    c64: []SCL.c64,

    pub fn init(slice: anytype) SliceUnion {
        return switch (UT.Child(@TypeOf(slice))) {
            SCL.q8 => .{ .q8 = slice },
            SCL.r16 => .{ .r16 = slice },
            SCL.r32 => .{ .r32 = slice },
            SCL.r64 => .{ .r64 = slice },
            SCL.c16 => .{ .c16 = slice },
            SCL.c32 => .{ .c32 = slice },
            SCL.c64 => .{ .c64 = slice },
            else => @compileError("Invalid Type for SliceUnion."),
        };
    }

    pub fn len(self: SliceUnion) SizeType {
        return switch (self) {
            inline else => |x| x.len,
        };
    }

    pub fn bytes(self: anytype) []u8 {
        return switch(self.*) {
             .q8 => std.mem.sliceAsBytes(self.q8),
            .r16 => std.mem.sliceAsBytes(self.r16),
            .r32 => std.mem.sliceAsBytes(self.r32),
            .r64 => std.mem.sliceAsBytes(self.r64),
            .c16 => std.mem.sliceAsBytes(self.c16),
            .c32 => std.mem.sliceAsBytes(self.c32),
            .c64 => std.mem.sliceAsBytes(self.c64),
        };
    }
};


pub fn computeStrides(sizes: Sizes, strides: []SizeType) void {
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

pub inline fn getSlice(comptime DataType: type, arg: SliceUnion) []DataType {
    return @field(arg, SCL.scalarName(DataType));
}
