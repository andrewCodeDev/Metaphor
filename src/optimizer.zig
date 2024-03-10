const SliceUnion = @import("tensor_components.zig").SliceUnion;
const SC = @import("scalar.zig");

pub const Optimizer = struct {
    const Self = @This();
    opt_ptr: *anyopaque,
    upd_ptr: *const fn(*anyopaque, []const u8, SliceUnion, SliceUnion) void,
    pub inline fn update(self: Self, name: []const u8, wgt: SliceUnion, grd: SliceUnion) void {
        self.upd_ptr(self.opt_ptr, name, wgt, grd);
    }
};

inline fn updateDispatch(
    opt: anytype,
    name: []const u8,
    wgt: SliceUnion,
    grd: SliceUnion) void {
    switch (wgt) {
        SC.r16 => opt.optimize(name, wgt.r16, grd.r16),
        SC.r32 => opt.optimize(name, wgt.r32, grd.r32),
        SC.r64 => opt.optimize(name, wgt.r64, grd.r64),
        else => {
             @panic("Optimizer: TODO");
        }
    }
}

pub const NullOptimizer = struct {

    const Self = @This();
    fn update(_: *anyopaque, _: []const u8, _: SliceUnion, _: SliceUnion) void {
        return;
    }    
    pub fn optimizer() Optimizer {
        return .{
            .opt_ptr = undefined, 
            .upd_ptr = Self.update,
        };
    }
};

//pub const SGDescent = struct {
//
//    const Self = @This();
//
//    rate: r16 = 0.01,
//
//    fn update(ptr: *anyopaque, _: []const u8, wgt: SliceUnion, grd: SliceUnion) void {
//        const self: *Self = @ptrCast(@alignCast(ptr));
//        @call(.always_inline, updateDispatch, .{ self, "", wgt, grd });
//    }
//
//    ///////////////////////////////////////////////////////////////////
//    
//    fn optimize(self: *Self, _: []const u8, wgt: anytype, grd: anytype) void {
//        LinAlg.simdOptimize(wgt, grd, self.rate);
//    }
//
//    pub fn optimizer(self: *Self) Optimizer {
//        return Optimizer {
//            .opt_ptr = self, 
//            .upd_ptr = Self.update,
//        };
//    }
//};
//
