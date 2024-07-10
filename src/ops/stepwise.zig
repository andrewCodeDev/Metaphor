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

    core.invoke(core.kernels.stepwise, core.dkey(z), .{
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

pub fn reverse(_: []const OpDatum) void {
    // TODO: this function has no derivative
}

pub fn derive(_: []const OpDatum, _: Tensor) ?OpDatum {
    return null;
}
