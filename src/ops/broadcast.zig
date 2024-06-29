const std = @import("std");
const core = @import("core");
const OpArgs = core.OpArgs;
const OpDatum = core.OpDatum;
const OpInterface = core.OpInterface;
const Tensor = core.Tensor;
const Graph = core.Graph;
const StaticStringMap = std.StaticStringMap;
const reduce = @import("reduce.zig");
const commute = @import("common.zig").commute;

// default to lhs-graph
pub fn forward(x: Tensor, sizes: []const usize, expr: []const u8) Tensor {
    return forward_impl(x.ptr, x, sizes, expr);
}

// enable choice of graph
pub fn forward_impl(graph: *Graph, x: Tensor, sizes: []const usize, expr: []const u8) Tensor {

    const op = forward_map.get(expr) orelse @panic("Invalid expression for reduction.");

    const y = op(graph, x, sizes);

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

pub fn derive(args: []const OpDatum, wrt: Tensor) ?OpDatum {

    const dx: OpDatum = core.derive(args[0].tensor, wrt) orelse return null;

    if (dx == .scalar) {

        // if we have scalars, we can't reverse
        // any further than that in the future,
        // and this tensor becomes an input.

        const u = wrt.ptr.tensor(.{ 
            .class = .inp, 
            .dtype = args[0].tensor.dtype(), 
            .sizes = args[0].tensor.sizes(),
        });

        const key = core.dkey(u);

        const factor: f64 = @floatFromInt(args[1].tensor.len() / args[0].tensor.len());

        core.kernels.fill[key](u.data_ptr(), dx.scalar * factor, u.len(), u.stream());

        return OpDatum{ .tensor = u };
    }

    // commute broadcast to reduce: i->ij : ij->i
    const expr = commute(args[0].expr, "->", wrt.ptr.node_arena.allocator());

    return OpDatum{ .tensor = reduce.forward_impl(wrt.ptr, dx.tensor, expr) };
}

//////////////////////////////
// Expression to Kernel Map //

const ForwardOp = *const fn (*Graph, Tensor, []const usize) Tensor;

const forward_map = std.StaticStringMap(ForwardOp).initComptime(.{
    .{ "i->ij", broadcast_i_ij },  
    .{ "j->ij", broadcast_j_ij },  
});

const ReverseOp = *const fn (Tensor, Tensor) void;

const reverse_map = std.StaticStringMap(ReverseOp).initComptime(.{
    .{ "i->ij", broadcast_i_ij_reverse },  
    .{ "j->ij", broadcast_j_ij_reverse },  
});

// <>--------------------------------------------------------<>

fn broadcast_i_ij(graph: *Graph, x: Tensor, sizes: []const usize) Tensor {

    std.debug.assert(x.rank() == 1);

    std.debug.assert(sizes.len == 2);

    const y = graph.tensor(.{
        .class = .hid,
        .dtype = x.dtype(),
        .sizes = sizes,
    });

    const key = core.dkey(y);

    core.kernels.broadcast_i_ij[key](x.data_ptr(), y.data_ptr(), 0.0, sizes[0], sizes[1], y.stream());

    return y;
}

fn broadcast_i_ij_reverse(x: Tensor, y: Tensor) void {

    const s = x.sizes();

    std.debug.assert(s.len == 2);

    const key = core.dkey(y);
    
    core.kernels.reduce_ij_i[key](y.grad_ptr(), x.grad_ptr(), 1.0, s[0], s[1], x.stream());
}

// <>--------------------------------------------------------<>

fn broadcast_j_ij(graph: *Graph, x: Tensor, sizes: []const usize) Tensor {

    std.debug.assert(x.rank() == 1);

    std.debug.assert(sizes.len == 2);

    const y = graph.tensor(.{
        .class = .hid,
        .dtype = x.dtype(),
        .sizes = sizes,
    });

    const key = core.dkey(y);

    core.kernels.broadcast_i_ij[key](x.data_ptr(), y.data_ptr(), 0.0, sizes[0], sizes[1], y.stream());

    return y;
}

fn broadcast_j_ij_reverse(x: Tensor, y: Tensor) void {

    const s = x.sizes();

    std.debug.assert(s.len == 2);

    const key = core.dkey(y);
    
    core.kernels.reduce_ij_j[key](y.grad_ptr(), x.grad_ptr(), 1.0, s[0], s[1], x.stream());
}
