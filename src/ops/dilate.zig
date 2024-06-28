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

    core.kernels.dilate[key](
        x.data_ptr(),
        value,
        z.data_ptr(),
        z.len(),
        z.stream()
    );

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
    
    core.kernels.dilate_reverse[key](
        args[2].tensor.grad_ptr(),
        args[1].scalar,
        args[0].tensor.grad_ptr(),
        args[0].tensor.len(),
        args[0].tensor.stream(),
    );
}

pub fn derive(args: []const OpDatum, wrt: Tensor) ?OpDatum {
    const dx = core.derive(args[0].tensor, wrt) orelse return null;

    if (dx == .scalar) 
        return OpDatum{ .scalar = dx.scalar * args[1].scalar };

    return OpDatum{ .tensor = forward_impl(wrt.ptr, dx.tensor, args[1].scalar) };
}
