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

    const key = core.dkey(z);

    core.invoke(core.kernels.relu, key, .{
        x.data_ptr(),
        z.data_ptr(),
        z.len(),
        z.context(),
    });

    if (graph.mode == .train) {
        core.attach_op(@This(), z, &.{ 
            OpDatum{ .tensor = x },
            OpDatum{ .tensor = z },
        });        
    }

    return z;
}

pub fn reverse(args: []const OpDatum) void {
    core.enable_gradient(args[0].tensor);

    const key = core.dkey(args[1].tensor);
    
    core.invoke(core.kernels.relu_reverse, key, .{
        args[0].tensor.data_ptr(),
        args[0].tensor.grad_ptr(),
        args[1].tensor.grad_ptr(),
        args[0].tensor.len(),
        args[0].tensor.context(),
    });
}

const stepwise = @import("stepwise.zig");

pub fn derive(args: []const OpDatum, wrt: Tensor) ?OpDatum {
    return OpDatum{ .tensor = stepwise.forward_impl(wrt.ptr, args[0].tensor) };
}
