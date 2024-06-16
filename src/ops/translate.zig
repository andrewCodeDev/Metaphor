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
    const z = graph.tensor(.hid, x.type_tag(), x.sizes());

    core.kernels.translate[z.type_id()](
        x.data_ptr(),
        value,
        z.data_ptr(),
        z.len(),
        z.stream()
    );

    if (graph.mode == .train) {

        const op = OpInterface.init(@This(), &.{ 
            OpDatum{ .tensor = x },
            OpDatum{ .scalar = value },
            OpDatum{ .tensor = z },
        });
        
        core.attach_op(op, z);
    }

    return z;
}

pub fn reverse(args: []const OpDatum, type_id: usize) void {
    core.enable_gradient(args[0].tensor);
    
    core.kernels.addition[type_id](
        args[2].tensor.grad_ptr(),
        args[0].tensor.grad_ptr(),
        args[0].tensor.grad_ptr(),
        args[0].tensor.len(),
        args[0].tensor.stream(),
    );
}

pub fn derive(args: []const OpDatum, wrt: Tensor) ?OpDatum {
    return core.derive(args[0].tensor, wrt);
}
