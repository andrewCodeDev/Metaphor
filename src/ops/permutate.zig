const std = @import("std");
const core = @import("core");
const OpArgs = core.OpArgs;
const OpDatum = core.OpDatum;
const OpInterface = core.OpInterface;
const Tensor = core.Tensor;
const Graph = core.Graph;

pub fn forward(x: Tensor, expr: []const u8) Tensor {
    return forward_impl(x.ptr, x, expr);
}

// enable choice of graph
pub fn forward_impl(graph: *Graph, x: Tensor, expr: []const u8) Tensor {

    const op = forward_map.get(expr) orelse @panic("Invalid expression for reduction.");

    const y = op(graph, x);

    if (graph.mode == .train) {
        core.attach_op(@This(), y, &.{ 
            OpDatum{ .tensor = x },
            OpDatum{ .tensor = y },
            OpDatum{ .expr = expr },
        });        
    }

    return y;
}

pub fn reverse(_: []const OpDatum) void {
    @panic("TODO");
}

pub fn derive(_: []const OpDatum, _: Tensor) ?OpDatum {
    @panic("TODO");
}

//////////////////////////////
// Expression to Kernel Map //

const ForwardOp = *const fn (*Graph, Tensor) Tensor;

const forward_map = std.StaticStringMap(ForwardOp).initComptime(.{
    .{ "ij->ji", ij_ji },  
});


// <>--------------------------------------------------------<>

fn ij_ji(graph: *Graph, x: Tensor) Tensor {

    const xs = x.sizes();

    std.debug.assert(xs.len == 2);

    const y = graph.tensor(.{
        .class = .hid,
        .dtype = x.dtype(),
        .sizes = &.{ xs[1], xs[0] },
    });

    const key = core.dkey(y);

    core.invoke(core.kernels.permutate_ij_ji, key, .{
        x.data_ptr(),
        y.data_ptr(),
        1.0, // alpha
        xs[0], xs[1],
        y.stream(),
    });

    return y;
}

// <>--------------------------------------------------------<>
