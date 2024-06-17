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

    const z = graph.tensor(.hid, x.type_tag(), x.sizes());

    core.kernels.stepwise[z.type_id()](
        x.data_ptr(),
        z.data_ptr(),
        z.len(),
        z.stream()
    );

    if (graph.mode == .train) {

        const op = OpInterface.init(@This(), &.{ 
            OpDatum{ .tensor = x },
            OpDatum{ .tensor = z },
        });
        
        core.attach_op(op, z);
    }

    return z;
}

pub fn reverse(args: []const OpDatum, _: usize) void {
    core.enable_gradient(args[0].tensor);    
    // TODO: this function only adds zero
}

pub fn derive(_: []const OpDatum, _: Tensor) ?OpDatum {
    return null;
}
