const std = @import("std");
const SC = @import("scalar.zig");
const UT = @import("utility.zig");
const CB = @import("callback_builder.zig");
const TC = @import("tensor_components.zig");
const DU = @import("device_utils.zig");
const TenOps = @import("tensor_ops.zig");
const Child = UT.Child;

const CallbackBuilder = CB.CallbackBuilder;
const CallbackDropReverse = CB.CallbackDropReverse;
const NoCleanup = CB.NoCleanup;
const NoArg = CB.NoArg;
const overloads = @import("kernel_overloads.zig");
const Parser = @import("expression_parsing.zig");
const Stream = DU.Stream;

/// MSE
pub fn mse(
    stream: Stream,
    src: anytype,
    trg: anytype,
    redux: ?[*]f32, // gpu
    score: ?[*]f32, // cpu
) void {
    const T = @TypeOf(src);

    const src_value = src.values();

    // if grads do not exist, we
    // assign to the src_value and
    // that is checked by the kernel
    var src_grads = src_value;

    if (src.grads()) |grads| {
        src_grads = grads;
    }

    const r = src.sizes().len;

    if (r == 2) {
        @panic("TODO: Implement mse Rank-2.");

        // copy to scratch memory
        // launch kernel
        // copy and return scalar
    } else if (r == 1) {
        // make a function for this - needs to be the same as grid.x
        const s_size: TC.SizeType = (src_value.len / (32 * 32 * 4)) + 2;

        const scratch = stream.getScratch(T.DataType, s_size);

        overloads.kernel_mse_loss_i_i.call(.{
            stream.context,
            src_value.ptr,
            src_grads.ptr,
            trg.values().ptr,
            scratch.ptr,
            redux,
            src_value.len,
        });
    } else {
        @panic("MSE requires Rank[1,2] tensors.");
    }

    if (score) |ptr| {
        DU.copyFromDevice(redux.?, @ptrCast(@alignCast(ptr)), stream);
        DU.synchronizeStream(stream);
    }
}

pub fn bce(
    stream: Stream,
    src: anytype,
    trg: anytype,
    redux: ?[*]f32, // gpu
    score: ?[*]f32, // cpu
) void {
    const T = @TypeOf(src);

    const src_value = src.values();

    // if grads do not exist, we
    // assign to the src_value and
    // that is checked by the kernel
    var src_grads = src_value;

    if (src.grads()) |grads| {
        src_grads = grads;
    }

    const r = src.sizes().len;

    if (r == 2) {
        @panic("TODO: Implement bce Rank-2.");

        // copy to scratch memory
        // launch kernel
        // copy and return scalar
    } else if (r == 1) {
        // make a function for this - needs to be the same as grid.x
        const s_size: TC.SizeType = (src_value.len / (32 * 32 * 4)) + 2;

        const scratch = stream.getScratch(T.DataType, s_size);

        overloads.kernel_sigmoid.call(.{
            stream.context, src.values().ptr, src.values().ptr, src.len()
        });

        overloads.kernel_bce_loss_i_i.call(.{
            stream.context,
            src_value.ptr,
            src_grads.ptr,
            trg.values().ptr,
            scratch.ptr,
            redux,
            src_value.len,
        });

    } else {
        @panic("MSE requires Rank[1,2] tensors.");
    }

    if (score) |ptr| {
        DU.copyFromDevice(redux.?, @ptrCast(@alignCast(ptr)), stream);
        DU.synchronizeStream(stream);
    }
}

/// CCE
pub fn cce(
    stream: Stream,
    src: anytype, // tensor input
    trg: anytype, // single integer or column of integers
    redux: ?[*]f32, // gpu
    score: ?[*]f32, // cpu
) void {
    const ST = @TypeOf(src);
    const TT = @TypeOf(trg);

    const src_value = src.values();

    // if grads do not exist, we
    // assign to the src_value and
    // that is checked by the kernel
    var src_grads = src_value;

    if (src.grads()) |grads| {
        src_grads = grads;
    }

    if (comptime UT.isSlice(TT)) {

        if (comptime !UT.isInteger(Child(TT))) {
            @compileError("CCE requires integer targets for each row.");
        }

        const src_sizes = src.sizes();

        std.debug.assert(src_sizes.len == 2);
        std.debug.assert(src_sizes[0] == trg.len);

        TenOps.softmaxForward_ij_j(stream, src, src);
        
        // make a function for this - needs to be the same as grid.x
        const s_size: TC.SizeType = (src_sizes[0] / 32) + 2;
        const scratch = stream.getScratch(ST.DataType, s_size);

        overloads.kernel_cce_loss_ij_j.call(.{
            stream.context, src_value.ptr, src_grads.ptr, trg.ptr, scratch.ptr, redux, src_sizes[0], src_sizes[1]
        });

        // copy to scratch memory
        // launch kernel
        // copy and return scalar
    } else if (comptime UT.isInteger(TT)) {
        // TODO: cleanup this size calculation.

        std.debug.assert(src.sizes().len == 1);

        // make a function for this - needs to be the same as grid.x
        const s_size: TC.SizeType = (src_value.len / (32 * 32 * 4)) + 2;
        const scratch = stream.getScratch(ST.DataType, s_size);

        overloads.kernel_softmax_i_i.call(.{ stream.context, src_value.ptr, src_value.ptr, scratch.ptr, src_value.len });
        overloads.kernel_cce_loss_i_i.call(.{ stream.context, src_value.ptr, src_grads.ptr, trg, scratch.ptr, redux, src_value.len });
    } else {
        @compileError("CCE requires either integer or slice-of-integers target.");
    }

    if (score) |ptr| {
        DU.copyFromDevice(redux.?, @ptrCast(@alignCast(ptr)), stream);

        DU.synchronizeStream(stream);
    }
}

// BCE
