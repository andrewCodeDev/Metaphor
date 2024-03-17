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



// MSE

// CCE
pub fn cce(
    stream: Stream,
    src: anytype,
    trg: anytype,
    redux: ?[*]f64, // gpu
    score: ?[*]f64, // cpu
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
        @compileError("TODO: Implement slice version.");

        // copy to scratch memory
        // launch kernel
        // copy and return scalar
    }  else if (comptime UT.isInteger(TT)) {

        // make a function for this - needs to be the same as grid.x
        const s_size: TC.SizeType = (src_value.len / (32 * 32 * 4)) + 2;

        const scratch = stream.getScratch(ST.DataType, s_size);

        overloads.kernel_softmax_i_i.call(.{
            stream.context, 
            src_value.ptr, 
            src_value.ptr, 
            scratch.ptr, 
            src_value.len
        });

        overloads.kernel_cce_loss_i_i.call(.{
            stream.context, 
            src_value.ptr,
            src_grads.ptr,
            trg, 
            scratch.ptr,
            redux,
            src_value.len,
        });

    } else {
        @compileError("CCE requires either integer or slice-of-integers target.");
    }

    if (score) |ptr| {

        DU.copyFromDevice(redux.?, @ptrCast(@alignCast(ptr)), stream);

        DU.synchronizeStream(stream);
    }
}

// BCE
