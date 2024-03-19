const SliceUnion = @import("tensor_components.zig").SliceUnion;
const SC = @import("scalar.zig");
const Stream = @import("device_utils.zig").Stream;
const overloads = @import("kernel_overloads.zig");
const std = @import("std");
const IndexType = @import("tensor_components.zig").IndexType;
const GraphID = usize; // convert graph ptr to number
const Child = @import("utility.zig").Child;

const ClipRange = struct {
    lower: f16,
    upper: f16,

    // clip at +/-inf bypasses clipping
    pub fn init() ClipRange {
        return .{
            .lower = -std.math.inf(f16),
            .upper = std.math.inf(f16),
        };
    }
};

// Some optimizers need to store extra data about the
// weight gradients (like momentum). We pass in both
// the GraphID and the current weight index to make
// hashing possible in the case of multiple graphs
// per optimizer.

pub const Optimizer = struct {
    opt_ptr: *anyopaque,
    upd_ptr: *const fn (
        *anyopaque,
        GraphID, // graph ptr value
        IndexType, // index of weight
        SliceUnion, // raw wgt values
        SliceUnion, // raw grd values
        Stream, // stream of wgt
    ) void,

    pub inline fn update(self: Optimizer, gid: GraphID, wid: IndexType, wgt: SliceUnion, grd: SliceUnion, stream: Stream) void {
        self.upd_ptr(self.opt_ptr, gid, wid, wgt, grd, stream);
    }
};

// helper to deduce dispatchable types... not strictly needed
inline fn updateDispatch(
    opt: anytype,
    gid: GraphID,
    wid: IndexType,
    wgt: SliceUnion,
    grd: SliceUnion,
    stream: Stream,
) void {
    switch (wgt) {
        .r16 => opt.optimize(gid, wid, wgt.r16, grd.r16, stream),
        .r32 => opt.optimize(gid, wid, wgt.r32, grd.r32, stream),
        .r64 => opt.optimize(gid, wid, wgt.r64, grd.r64, stream),
        .c16 => opt.optimize(gid, wid, wgt.r16, grd.r16, stream),
        .c32 => opt.optimize(gid, wid, wgt.r32, grd.r32, stream),
        .c64 => opt.optimize(gid, wid, wgt.r64, grd.r64, stream),
        else => @panic("Optimizer: TODO - q8"),
    }
}

pub const NullOptimizer = struct {

    // "It's a show about nothing! It does nothing... but it does it in style!"
    //     ~ Andrei Alexandrescu

    fn update(_: *anyopaque, _: GraphID, _: IndexType, _: SliceUnion, _: SliceUnion, _: Stream) void {
        return {};
    }

    pub fn optimizer() Optimizer {
        return .{
            .opt_ptr = undefined,
            .upd_ptr = NullOptimizer.update,
        };
    }
};

pub const SGD = struct {
    rate: f16,
    clip: ClipRange,

    pub fn init(config: struct {
        rate: f16,
        clip: ?ClipRange = null,
    }) SGD {
        return .{
            .rate = config.rate,
            .clip = config.clip orelse ClipRange.init(),
        };
    }

    fn update(ptr: *anyopaque, gid: GraphID, wid: IndexType, wgt: SliceUnion, grd: SliceUnion, stream: Stream) void {
        const self: *SGD = @ptrCast(@alignCast(ptr));
        @call(.always_inline, updateDispatch, .{ self, gid, wid, wgt, grd, stream });
    }

    ///////////////////////////////////////////////////////////////////

    fn optimize(
        self: *SGD,
        _: GraphID,
        _: IndexType,
        wgt: anytype,
        grd: anytype,
        stream: Stream,
    ) void {
        std.debug.assert(wgt.len == grd.len);
        const T = Child(@TypeOf(grd));
        overloads.kernel_gradient_descent.call(.{
            stream.context,
            wgt.ptr,
            grd.ptr,
            SC.asScalar(T, self.rate),
            SC.asScalar(T, self.clip.lower),
            SC.asScalar(T, self.clip.upper),
            wgt.len,
        });
    }

    pub fn optimizer(self: *SGD) Optimizer {
        return .{
            .opt_ptr = self,
            .upd_ptr = SGD.update,
        };
    }
};
