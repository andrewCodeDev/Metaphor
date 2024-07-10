const std = @import("std");
const core = @import("core");
const OpArgs = core.OpArgs;
const OpDatum = core.OpDatum;
const OpInterface = core.OpInterface;
const Tensor = core.Tensor;
const Graph = core.Graph;

pub fn forward(x: Tensor) Tensor {
    return forward_impl(x.ptr, x);
}

pub fn forward_impl(graph: *Graph, x: Tensor) Tensor {

    const z = graph.tensor(.{ 
        .class = .hid,
        .dtype = x.dtype(), 
        .sizes = x.sizes(), 
    });

    core.invoke(core.kernels.negate, core.dkey(z), .{
        x.data_ptr(),
        z.data_ptr(),
        z.len(),
        z.context(),
    });

    if (graph.mode == .train) {

        const op = OpInterface.init(@This(), &.{ 
            OpDatum{ .tensor = x },
            OpDatum{ .tensor = z },
        });
        
        core.attach_op(op, z);
    }

    return z;
}

pub fn reverse(args: []const OpDatum) void {
    core.enable_gradient(args[0].tensor);

    const key = core.dkey(args[1].tensor);
    
    core.invoke(core.kernels.subtraction, key, .{
        args[0].tensor.grad_ptr(),
        args[1].tensor.grad_ptr(),
        args[0].tensor.grad_ptr(),
        args[0].tensor.len(),
        args[0].tensor.context(),
    });
}

pub fn derive(args: []const OpDatum, wrt: Tensor) ?OpDatum {
    const dx = core.derive(args[0].tensor, wrt) orelse return null;
    return switch (dx) {
        .scalar => |s| OpDatum{ .scalar = -s },
        .tensor => |t| OpDatum{ .tensor = forward_impl(wrt.ptr, t) },
    };
}
