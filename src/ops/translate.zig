const std = @import("std");
const core = @import("core");
const OpArgs = core.OpArgs;
const OpDatum = core.OpDatum;
const OpInterface = core.OpInterface;
const Tensor = core.Tensor;
const Graph = core.Graph;

pub fn forward(x: Tensor, value: f64) Tensor {
    return forward_impl(x.ptr, x, value);
}

pub fn forward_impl(
    graph: *Graph,
    x: Tensor, 
    value: f64, 
) Tensor {

    const z = graph.tensor(.{ 
        .class = .hid, 
        .dtype = x.dtype(),
        .sizes = x.sizes(),
    });

    const key = core.dkey(z);

    core.invoke(core.kernels.translate, key, .{
        x.data_ptr(),
        value,
        z.data_ptr(),
        z.len(),
        z.context(),
    });

    if (graph.mode == .train) {
        core.attach_op(@This(), z, &.{ 
            OpDatum{ .tensor = x },
            OpDatum{ .scalar = value },
            OpDatum{ .tensor = z },
        });
    }
    return z;
}

pub fn reverse(args: []const OpDatum) void {

    core.enable_gradient(args[0].tensor);

    const key = core.dkey(args[0].tensor);
    
    core.invoke(core.kernels.addition, key, .{
        args[2].tensor.grad_ptr(),
        args[0].tensor.grad_ptr(),
        args[0].tensor.grad_ptr(),
        args[0].tensor.len(),
        args[0].tensor.context(),
    });
}

pub fn derive(args: []const OpDatum, wrt: Tensor) ?OpDatum {
    return core.derive(args[0].tensor, wrt);
}
