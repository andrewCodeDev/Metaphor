const std = @import("std");
const core = @import("core");
const OpArgs = core.OpArgs;
const OpDatum = core.OpDatum;
const OpInterface = core.OpInterface;
const Tensor = core.Tensor;
const Graph = core.Graph;
const com = @import("common.zig");

pub fn forward(x: Tensor, y: Tensor, expr: []const u8) Tensor {
    return forward_scaled(x, y, 1.0, expr);
}

// default to lhs-graph
pub fn forward_scaled(x: Tensor, y: Tensor, scale: f64, expr: []const u8) Tensor {
    return forward_impl(x.ptr, x, y, scale, expr);
}

pub fn deduce_output(graph: *Graph, x: Tensor, y: Tensor, expr: []const u8) Tensor {

    std.debug.assert(x.dtype() == y.dtype());

    const comma = std.mem.indexOfScalar(u8, expr, ',') orelse unreachable;
    const arrow = com.arrow_position(expr);

    const lhs = expr[0..comma];
    const rhs = expr[comma+1..arrow.head];
    const out = expr[arrow.tail+1..];
    
    std.debug.assert(x.rank() == lhs.len);
    std.debug.assert(y.rank() == rhs.len);
    // TODO: this number was picked arbitrarily
    std.debug.assert(out.len <= 8);

    var sizes: [8]usize = undefined;

    for (out, 0..) |o, i| {   
        for (x.sizes(), lhs) |n, c| { if (o == c) sizes[i] = n; }
    }
    
    for (out, 0..) |o, i| {   
        for (y.sizes(), rhs) |n, c| { if (o == c) sizes[i] = n; }
    }

    return graph.tensor(.{
        .class = .hid,
        .dtype = x.dtype(),
        .sizes = sizes[0..out.len],
    });
}

// enable choice of graph
pub fn forward_impl(   
    graph: *Graph,
    x: Tensor, 
    y: Tensor,
    scale: f64, 
    expr: []const u8,
) Tensor {

    const z = deduce_output(graph, x, y, expr);

    get_forward(expr)(
        graph, core.dkey(x), 
        x.data_ptr(), x.sizes(), x.len(),
        y.data_ptr(), y.sizes(), y.len(),
        scale,
        z.data_ptr(),
        0.0,
    );

    if (graph.mode == .train) {
        core.attach_op(@This(), z, &.{ 
            OpDatum{ .tensor = x },
            OpDatum{ .tensor = y },
            OpDatum{ .tensor = z },
            OpDatum{ .scalar = scale },
            OpDatum{ .expr = expr },
        });        
    }

    return z;
}

pub fn reverse(args: []const OpDatum) void {
    get_reverse(args[4].expr)(args[0].tensor, args[1].tensor, args[2].tensor, args[3].scalar);
}

pub fn derive(_: []const OpDatum, _: Tensor) ?OpDatum {
}

//////////////////////////////
// Expression to Kernel Map //

// forward ops need to be very generic because we
// will pass the gradient pointer at different times
// instead of the data pointer circumstantially

const ForwardOp = *const fn (
    *Graph, 
    usize, // key
    ?*anyopaque, []const usize, usize, // x tensor
    ?*anyopaque, []const usize, usize, // y tensor
    f64, // alpha
    ?*anyopaque, // output
    f64, // beta
) void;

const ReverseOp = *const fn (Tensor, Tensor, Tensor, f64) void;

const Pair = struct {
    fwd: ForwardOp,
    rev: ReverseOp,
};

const product_map = std.StaticStringMap(Pair).initComptime(.{
    .{ "i,ij->j", Pair{ .fwd = i_ij_j, .rev = i_ij_j_reverse } },  
    .{ "ij,j->i", Pair{ .fwd = ij_j_i, .rev = ij_j_i_reverse } },  
    .{ "ij,jk->ik", Pair{ .fwd = ij_jk_ik, .rev = ij_jk_ik_reverse } },  
    .{ "ij,kj->ik", Pair{ .fwd = ij_kj_ik, .rev = ij_kj_ik_reverse } },  
    .{ "ji,jk->ik", Pair{ .fwd = ji_jk_ik, .rev = ji_jk_ik_reverse } },  
});

inline fn get_forward(expr: []const u8) ForwardOp {
    const pair = product_map.get(expr) orelse @panic("Invalid expression for inner product.");
    return pair.fwd;
}

inline fn get_reverse(expr: []const u8) ReverseOp {
    const pair = product_map.get(expr) orelse @panic("Invalid expression for inner product.");
    return pair.rev;
}

// <>--------------------------------------------------------<>

fn ij_j_i(
    graph: *Graph, 
    key: usize,
    xd: ?*anyopaque, xs: []const usize, xl: usize,
    yd: ?*anyopaque, ys: []const usize, yl: usize,
    alpha: f64,
    zd: ?*anyopaque,
    beta: f64,
) void {

    std.debug.assert(xs.len == 2);
    std.debug.assert(ys.len == 1);
    std.debug.assert(xs[1] == ys[0]);

    // edtend y by one
    const es: []const usize = &.{ ys[0], 1 };

    ij_jk_ik(
        graph, key,
        xd, xs, xl,
        yd, es, yl,
        alpha,
        zd,
        beta
    );
}

fn ij_j_i_reverse(x: Tensor, y: Tensor, z: Tensor, scale: f64) void {

    core.enable_gradient(x);
    core.enable_gradient(y);

    // ex: (2,3)(3,1)->(2,1)

    // suppose a transpose to row-wise vector
    const y_rowwise: []const usize = &.{ 1, y.sizes()[0] };
    const z_colwise: []const usize = &.{ z.sizes()[0], 1 };

    ij_jk_ik( // dx: (2,1)(1,3)->(2,3): G(z).T(y)
        x.ptr, core.dkey(x),
        z.grad_ptr(), z_colwise, z.len(),
        y.data_ptr(), y_rowwise, y.len(),
        scale,
        x.grad_ptr(),
        1.0,
    );

    ji_jk_ik( // dy: (3,2)(2,1)->(3,1): T(x).G(z)
        y.ptr, core.dkey(y),
        x.data_ptr(), x.sizes(), x.len(),
        z.grad_ptr(), z_colwise, z.len(),
        scale,
        y.grad_ptr(),
        1.0,
    );
}

// <>--------------------------------------------------------<>

fn i_ij_j(
    graph: *Graph, 
    key: usize,
    xd: ?*anyopaque, xs: []const usize, xl: usize,
    yd: ?*anyopaque, ys: []const usize, yl: usize,
    alpha: f64,
    zd: ?*anyopaque,
    beta: f64,
) void {

    std.debug.assert(xs.len == 1);
    std.debug.assert(ys.len == 2);
    std.debug.assert(xs[0] == ys[0]);

    // edtend x by one
    const es: []const usize = &.{ 1, xs[0] };

    ij_jk_ik(
        graph, key,
        xd, es, xl,
        yd, ys, yl,
        alpha,
        zd,
        beta
    );
}

fn i_ij_j_reverse(x: Tensor, y: Tensor, z: Tensor, scale: f64) void {

    core.enable_gradient(x);
    core.enable_gradient(y);

    // ex: (1,3)(3,2)->(1,2)

    ij_kj_ik( // dx: (1,2)(2,3)->(1,3): G(z).T(y)
        x.ptr,
        core.dkey(x),
        z.grad_ptr(), z.sizes(), z.len(),
        y.data_ptr(), y.sizes(), y.len(),
        scale,
        x.grad_ptr(),
        1.0,
    );

    // edtend x by one - faux transposition
    const x_colwise: []const usize = &.{ x.sizes()[0], 1 };

    ij_jk_ik( // dy: (3,1)(1,2)->(3,2): T(x).G(z)
        y.ptr,
        core.dkey(y),
        x.data_ptr(), x_colwise, x.len(),
        z.grad_ptr(), z.sizes(), z.len(),
        scale,
        y.grad_ptr(),
        1.0,
    );
}

// <>--------------------------------------------------------<>


fn ij_jk_ik(
    graph: *Graph, 
    key: usize,
    xd: ?*anyopaque, xs: []const usize, _: usize,
    yd: ?*anyopaque, ys: []const usize, _: usize,
    alpha: f64,
    zd: ?*anyopaque,
    beta: f64,
) void {

    std.debug.assert(xs.len == 2);
    std.debug.assert(ys.len == 2);
    std.debug.assert(xs[1] == ys[0]);

    core.invoke(core.kernels.inner_product_ij_jk_ik, key, .{
        xd, yd, alpha, zd, beta, xs[0], xs[1], ys[1], graph.stream.context,
    });
}

fn ij_jk_ik_reverse(x: Tensor, y: Tensor, z: Tensor, scale: f64) void {

    core.enable_gradient(x);
    core.enable_gradient(y);

    // ex: (2,3)(3,4)->(2,4)

    ij_kj_ik( // dx: (2,4)(4,3)->(2,3): G(z).T(y)
        x.ptr,
        core.dkey(x),
        z.grad_ptr(), z.sizes(), z.len(),
        y.data_ptr(), y.sizes(), y.len(),
        scale,
        x.grad_ptr(),
        1.0,
    );

    ji_jk_ik( // dy: (3,2)(2,4)->(3,4): T(x).G(z)
        y.ptr,
        core.dkey(y),
        x.data_ptr(), x.sizes(), x.len(),
        z.grad_ptr().?, z.sizes(), z.len(),
        scale,
        y.grad_ptr(),
        1.0,
    );
}

// <>--------------------------------------------------------<>
    
fn ij_kj_ik(
    graph: *Graph,
    key: usize,
    xd: ?*anyopaque, xs: []const usize, _: usize,
    yd: ?*anyopaque, ys: []const usize, yl: usize,
    alpha: f64,
    zd: ?*anyopaque,
    beta: f64,
) void {

    std.debug.assert(xs.len == 2);
    std.debug.assert(ys.len == 2);
    std.debug.assert(xs[1] == ys[1]);

    // get scratch to perform transposition on y
    const td = graph.stream.get_scratch(@enumFromInt(key), yl);

    // transpose to the scratch memory
    core.invoke(core.kernels.permutate_ij_ji, key, .{
        yd, td, 0.0, ys[0], ys[1], graph.stream.context,
    });

    core.invoke(core.kernels.inner_product_ij_jk_ik, key, .{
        xd, td, alpha, zd, beta, xs[0], xs[1], ys[0], graph.stream.context,
    });
}

fn ij_kj_ik_reverse(x: Tensor, y: Tensor, z: Tensor, scale: f64) void {

    core.enable_gradient(x);
    core.enable_gradient(y);

    // ex: (2,3)(4,3)->(2,4), 

    // dx: (2,4)(4,3)->(2,3), G(z).y
    ij_jk_ik(
        x.ptr,
        core.dkey(x),
        z.grad_ptr(), z.sizes(), z.len(),
        y.data_ptr(), y.sizes(), y.len(),
        scale,
        x.grad_ptr(),
        1.0,
    );
    
    // dy: (4,2)(2,3)->(4,3), G(z).y
    ji_jk_ik(
        y.ptr,
        core.dkey(y),
        z.grad_ptr(), z.sizes(), z.len(),
        x.data_ptr(), x.sizes(), x.len(),
        scale,
        y.grad_ptr(),
        1.0,
    );
}

// <>--------------------------------------------------------<>

fn ji_jk_ik(
    graph: *Graph,
    key: usize,
    xd: ?*anyopaque, xs: []const usize, xl: usize,
    yd: ?*anyopaque, ys: []const usize,  _: usize,
    alpha: f64,
    zd: ?*anyopaque,
    beta: f64,
) void {

    std.debug.assert(xs.len == 2);
    std.debug.assert(ys.len == 2);
    std.debug.assert(xs[0] == ys[0]);

    // get scratch to perform transposition
    const tx = graph.stream.get_scratch(@enumFromInt(key), xl);

    // transpose to the scratch memory
    core.invoke(core.kernels.permutate_ij_ji, key, .{
        xd, tx, 0.0, xs[0], xs[1], graph.stream.context,
    });

    core.invoke(core.kernels.inner_product_ij_jk_ik, key, .{
        tx, yd, alpha, zd, beta, xs[1], xs[0], ys[1], graph.stream.context,
    });    
}

fn ji_jk_ik_reverse(x: Tensor, y: Tensor, z: Tensor, scale: f64) void {

    core.enable_gradient(x);
    core.enable_gradient(y);

    // ex: (3,2)(3,4)->(2,4)

    // dx: (3,4)(4,2)->(3,2), y.T(G(z))
    ij_kj_ik(
        x.ptr,
        core.dkey(x),
        y.data_ptr(), y.sizes(), y.len(),
        z.grad_ptr(), z.sizes(), z.len(),
        scale,
        x.grad_ptr(),
        1.0,
    );

    // dy: (3,2)(2,4)->(3,4), y.G(x)
    ij_jk_ik(
        y.ptr,
        core.dkey(y),
        x.data_ptr(), x.sizes(), x.len(),
        z.grad_ptr(), z.sizes(), z.len(),
        scale,
        y.grad_ptr(),
        1.0,
    );
}

// <>--------------------------------------------------------<>
