
const std = @import("std");
const assert = std.debug.assert;
const SCL = @import("scalar.zig");
const UT = @import("utility.zig");

pub const IndexType = usize;
pub const SizeType = usize;
pub const Strides = []const SizeType;
pub const Sizes = []const SizeType;

pub fn computeStrides(sizes: Sizes, strides: []SizeType) void {

    assert(sizes.len == strides.len);
    
    if(sizes.len == 1) {
        strides[0] = 1;
        return;
    }
    var i: usize = (sizes.len - 1);
    var n: SizeType = 1;

    while(i > 0) : (i -= 1) {
        strides[i] = n;
        n *= sizes[i];
    }
    strides[0] = n;
}

pub const SliceUnion = union(enum) {
    const Self = @This();
    
     q8: []SCL.q8,  
    r16: []SCL.r16,  
    r32: []SCL.r32,  
    r64: []SCL.r64,  
    c16: []SCL.c16, 
    c32: []SCL.c32, 
    c64: []SCL.c64, 

    pub fn len(self: Self) SizeType {
        return switch (self) {
            inline else => |x| x.len,  
        };
    }

    pub fn init(slice: anytype) Self {
        const T = UT.Child(@TypeOf(slice));
        
        return switch (T) {
             SCL.q8 => Self{  .q8 = slice },  
            SCL.r16 => Self{ .r16 = slice },  
            SCL.r32 => Self{ .r32 = slice },  
            SCL.r64 => Self{ .r64 = slice },  
            SCL.c16 => Self{ .c16 = slice },  
            SCL.c32 => Self{ .c32 = slice },  
            SCL.c64 => Self{ .c64 = slice },  
            else => @compileError("Invalid Type for SliceUnion."),        
        };
    }
};

pub inline fn getSlice(comptime DataType: type, arg: SliceUnion) []DataType {
    return @field(arg, SCL.scalarName(DataType));
}

