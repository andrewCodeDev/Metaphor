const std = @import("std");
const core = @import("core");
const OpArgs = core.OpArgs;
const OpDatum = core.OpDatum;
const OpInterface = core.OpInterface;
const Tensor = core.Tensor;
const Graph = core.Graph;
const com = @import("common.zig");

pub fn forward(x: Tensor, expr: []const u8) Tensor {
    return forward_impl(x.ptr, x, expr);
}

pub fn deduce_output(graph: *Graph, x: Tensor, expr: []const u8) Tensor {

    const arrow = com.arrow_position(expr);

    const lhs = expr[0..arrow.head];
    const rhs = expr[arrow.tail+1..];
    
    std.debug.assert(x.rank() == lhs.len);
    // TODO: this number was picked arbitrarily
    std.debug.assert(rhs.len <= 8);

    var sizes: [8]usize = undefined;

    for (rhs, 0..) |r, i| {   
        for (x.sizes(), lhs) |n, c| { if (r == c) sizes[i] = n; }
    }

    return graph.tensor(.{
        .class = .hid,
        .dtype = x.dtype(),
        .sizes = sizes[0..rhs.len],
    });
}

// enable choice of graph
pub fn forward_impl(graph: *Graph, x: Tensor, expr: []const u8) Tensor {

    const op = permutate_map.get(expr) orelse @panic("Invalid expression for permutate.");

    const y = deduce_output(graph, x, expr);

    op(graph, core.dkey(x), x.data_ptr(), x.sizes(), 0.0, y.data_ptr());

    if (graph.mode == .train) {
        core.attach_op(@This(), y, &.{ 
            OpDatum{ .tensor = x },
            OpDatum{ .tensor = y },
            OpDatum{ .expr = expr },
        });        
    }

    return y;
}

// the reverse of a permutation is the same perumtation
pub fn reverse(args: []const OpDatum) void {

    const x = args[0].tensor;
    const y = args[1].tensor;

    core.enable_gradient(x);

    const op = permutate_map.get(args[2].expr) orelse @panic("Invalid expression for permutate.");

    op(y.ptr, core.dkey(x), y.grad_ptr(), y.sizes(), 1.0, x.grad_ptr());
}

pub fn derive(_: []const OpDatum, _: Tensor) ?OpDatum {
    @panic("TODO");
}

//////////////////////////////
// Expression to Kernel Map //

const ForwardOp = *const fn (
    *Graph, 
    usize, // key
    ?*anyopaque, []const usize, // x tensor
    f64, // alpha
    ?*anyopaque, // output
) void;

const permutate_map = std.StaticStringMap(ForwardOp).initComptime(.{
    .{ "ij->ji", ij_ji },  
});

// <>--------------------------------------------------------<>

fn ij_ji(
    graph: *Graph, 
    key: usize,
    xd: ?*anyopaque, xs: []const usize,
    alpha: f64,
    yd: ?*anyopaque,
) void {
    std.debug.assert(xs.len == 2);

    core.invoke(core.kernels.permutate_ij_ji, key, .{
        xd, yd, alpha, xs[0], xs[1], graph.stream.context,
    });
}

// <>--------------------------------------------------------<>
