const std = @import("std");
const core = @import("core");
const OpArgs = core.OpArgs;
const OpDatum = core.OpDatum;
const OpInterface = core.OpInterface;
const Tensor = core.Tensor;
const Graph = core.Graph;
const permutate = @import("permutate.zig");

pub fn forward(x: Tensor, y: Tensor, expr: []const u8) Tensor {
    return forward_scaled(x, y, 1.0, expr);
}

// default to lhs-graph
pub fn forward_scaled(x: Tensor, y: Tensor, scale: f64, expr: []const u8) Tensor {
    return forward_impl(x.ptr, x, y, scale, expr);
}

// enable choice of graph
pub fn forward_impl(   
    graph: *Graph,
    x: Tensor, 
    y: Tensor,
    scale: f64, 
    expr: []const u8,
) Tensor {

    std.debug.assert(x.dtype() == y.dtype());

    const op = forward_map.get(expr) orelse @panic("Invalid expression for reduction.");

    const z = op(graph, x, y, scale);

    if (graph.mode == .train) {
        core.attach_op(@This(), z, &.{ 
            OpDatum{ .tensor = x },
            OpDatum{ .tensor = y },
            OpDatum{ .scalar = scale },
            OpDatum{ .expr = expr },
            OpDatum{ .tensor = z },
        });        
    }

    return z;
}

pub fn reverse(_: []const OpDatum) void {
    @panic("TODO");
}

pub fn derive(_: []const OpDatum, _: Tensor) ?OpDatum {
    @panic("TODO");
}

//////////////////////////////
// Expression to Kernel Map //

const ForwardOp = *const fn (*Graph, Tensor, Tensor, f64) Tensor;

const forward_map = std.StaticStringMap(ForwardOp).initComptime(.{
    .{ "ij,jk->ik", ij_jk_ik },  
    .{ "ij,kj->ik", ij_kj_ik },  
});


// <>--------------------------------------------------------<>

fn ij_jk_ik(graph: *Graph, x: Tensor, y: Tensor, scale: f64) Tensor {

    const xs = x.sizes();
    const ys = y.sizes();

    std.log.info("sizes - {any}", .{ ys });

    std.debug.assert(xs.len == 2);
    std.debug.assert(ys.len == 2);
    std.debug.assert(xs[1] == ys[0]);

    const z = graph.tensor(.{
        .class = .hid,
        .dtype = x.dtype(),
        .sizes = &.{ xs[0], ys[1] },
    });

    const key = core.dkey(z);

    core.kernels.inner_product_ij_jk_ik[key](
        x.data_ptr(),
        y.data_ptr(),
        scale, // alpha
        z.data_ptr(),
        0.0, // beta
        xs[0], xs[1], ys[1],
        z.stream(),
    );
    
    return z;
}

// <>--------------------------------------------------------<>
    
fn ij_kj_ik(graph: *Graph, x: Tensor, y: Tensor, scale: f64) Tensor {

    // TODO: consider making this in situ
    const t = permutate.forward_impl(graph, y, "ij->ji");

    return ij_jk_ik(graph, x, t, scale);
}

// <>--------------------------------------------------------<>
    
fn ji_jk_ik(graph: *Graph, x: Tensor, y: Tensor, scale: f64) Tensor {

    // TODO: consider making this in situ
    const t = permutate.forward_impl(graph, x, "ij->ji");

    return ij_jk_ik(graph, t, y, scale);
}

// <>--------------------------------------------------------<>
