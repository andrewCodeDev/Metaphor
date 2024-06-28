
const std = @import("std");
const core = @import("core");
const OpArgs = core.OpArgs;
const OpDatum = core.OpDatum;
const OpInterface = core.OpInterface;
const Tensor = core.Tensor;
const Graph = core.Graph;
const is_one = @import("common.zig").is_one;
const is_zero = @import("common.zig").is_zero;
const negate = @import("negate.zig");
const translate = @import("translate.zig");

// default to lhs-graph
pub fn forward(x: Tensor, y: Tensor) Tensor {
    return forward_impl(x.ptr, x, y);
}

fn deduce_output(graph: *Graph, x: Tensor) Tensor {
    return graph.tensor(.{ 
        .class = .hid, 
        .dtype = x.dtype(), 
        .sizes = x.sizes(),
    });
}

// enable choice of graph
pub fn forward_impl(   
    graph: *Graph,
    x: Tensor, 
    y: Tensor, 
) Tensor {

    std.debug.assert(x.dtype() == y.dtype());

    std.debug.assert(std.mem.eql(usize, x.sizes(), y.sizes()));

    const z = deduce_output(graph, x);

    const key = core.dkey(z);

    core.kernels.subtraction[key](
        x.data_ptr(), 
        y.data_ptr(), 
        z.data_ptr(), 
        z.len(), 
        z.stream(),
    );        

    if (graph.mode == .train) {
        core.attach_op(@This(), z, &.{ 
            OpDatum{ .tensor = x },
            OpDatum{ .tensor = y },
            OpDatum{ .tensor = z },
        });        
    }

    return z;
}

pub fn reverse(args: []const OpDatum) void {

    core.enable_gradient(args[0].tensor);
    core.enable_gradient(args[1].tensor);

    const key = core.dkey(args[2].tensor);
    
    core.kernels.addition[key](
        args[2].tensor.grad_ptr(),
        args[0].tensor.grad_ptr(),
        args[0].tensor.grad_ptr(),
        args[0].tensor.len(),
        args[0].tensor.stream(),
    );

    core.kernels.subtraction[key](
        args[2].tensor.grad_ptr(),
        args[1].tensor.grad_ptr(),
        args[1].tensor.grad_ptr(),
        args[1].tensor.len(),
        args[1].tensor.stream(),
    );
}

pub fn derive(
    args: []const OpDatum,
    wrt: Tensor,
) ?OpDatum {

    const dx: OpDatum = core.derive(args[0].tensor, wrt) orelse {

        const dy = core.derive(args[1].tensor, wrt) orelse return null;
        
        return switch (dy) {
            .scalar => |s| OpDatum{ .scalar = -s },
            .tensor => |t| OpDatum{ .tensor = negate.forward_impl(wrt.ptr, t) },
        };
    };

    const dy: OpDatum = core.derive(args[1].tensor, wrt) orelse {
        return dx;
    };

    // ex: x + x -> 1 + 1
    if (dx == .scalar and dy == .scalar)
        return OpDatum{ .scalar = dx.scalar - dy.scalar };

    // c + g'(x)
    if (dx == .scalar) {
        // TODO: this could be more optimal
        const u = negate.forward_impl(wrt.ptr, dy.tensor);
        if (is_zero(dx.scalar)) return OpDatum{ .tensor = u };
        return OpDatum{ .tensor = translate.forward_impl(wrt.ptr, u, dx.scalar) };
    }

    // f'(x) - c
    if (dy == .scalar) {
        if (is_zero(dy.scalar)) return dx;
        return OpDatum{ .tensor = translate.forward_impl(wrt.ptr, dx.tensor, -dy.scalar) };
    }

    // f'(x) - g'(x)
    return OpDatum{ .tensor = forward_impl(wrt.ptr, dx.tensor, dy.tensor) };
}
