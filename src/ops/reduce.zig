const std = @import("std");
const core = @import("core");
const OpArgs = core.OpArgs;
const OpDatum = core.OpDatum;
const OpInterface = core.OpInterface;
const Tensor = core.Tensor;
const Graph = core.Graph;
const StaticStringMap = std.StaticStringMap;

// default to lhs-graph
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

pub fn reverse(args: []const OpDatum) void {

    core.enable_gradient(args[0].tensor);

    const op = reverse_map.get(args[2].expr) orelse @panic("Invalid expression for reduction.");

    op(args[0].tensor, args[1].tensor);
}

pub fn derive(_: []const OpDatum, _: Tensor) ?OpDatum {

    @panic("TODO: Not Implemented");

//    const dx: OpDatum = core.derive(args[0].tensor, wrt) orelse return null;
//    return OpDatum{ .tensor = forward_impl(wrt.ptr, dx.tensor, dy.tensor) };
}

//////////////////////////////
// Expression to Kernel Map //

const ForwardOp = *const fn (*Graph, Tensor) Tensor;

const forward_map = std.StaticStringMap(ForwardOp).initComptime(.{
    .{ "ij->i", reduce_ij_i },  
    .{ "ij->j", reduce_ij_j },  
});

const ReverseOp = *const fn (Tensor, Tensor) void;

const reverse_map = std.StaticStringMap(ReverseOp).initComptime(.{
    .{ "ij->i", reduce_ij_i_reverse },  
    .{ "ij->j", reduce_ij_j_reverse },  
});

// <>--------------------------------------------------------<>

fn reduce_ij_i(graph: *Graph, x: Tensor) Tensor {

    const s = x.sizes();

    std.debug.assert(s.len == 2);

    const y = graph.tensor(.{
        .class = .hid,
        .dtype = x.dtype(),
        .sizes = s[0..1],
    });

    const key = core.dkey(y);

    core.kernels.reduce_ij_i[key](x.data_ptr(), y.data_ptr(), 0.0, s[0], s[1], y.stream());

    return y;
}

fn reduce_ij_i_reverse(x: Tensor, y: Tensor) void {

    const s = x.sizes();

    std.debug.assert(s.len == 2);

    const key = core.dkey(y);
    
    core.kernels.broadcast_i_ij[key](y.grad_ptr(), x.grad_ptr(), 1.0, s[0], s[1], x.stream());
}

// <>--------------------------------------------------------<>

fn reduce_ij_j(graph: *Graph, x: Tensor) Tensor {

    const s = x.sizes();

    std.debug.assert(s.len == 2);

    const y = graph.tensor(.{
        .class = .hid,
        .dtype = x.dtype(),
        .sizes = s[1..],
    });

    const key = core.dkey(y);

    core.kernels.reduce_ij_j[key](x.data_ptr(), y.data_ptr(), 0.0, s[0], s[1], y.stream());

    return y;
}

fn reduce_ij_j_reverse(x: Tensor, y: Tensor) void {

    const s = x.sizes();

    std.debug.assert(s.len == 2);

    const key = core.dkey(y);
    
    core.kernels.broadcast_j_ij[key](y.grad_ptr(), x.grad_ptr(), 1.0, s[0], s[1], x.stream());
}
