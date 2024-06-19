const std = @import("std");
const core = @import("core");
const OpArgs = core.OpArgs;
const OpDatum = core.OpDatum;
const OpInterface = core.OpInterface;
const Tensor = core.Tensor;
const Graph = core.Graph;
const is_one = @import("common.zig").is_one;
const is_zero = @import("common.zig").is_zero;

pub fn forward(x: Tensor, y: Tensor) Tensor {
    return forward_impl(x.ptr, x, y);
}

pub fn forward_impl(
    graph: *Graph,
    x: Tensor, 
    y: Tensor, 
) Tensor {
    std.debug.assert(x.type_id() == y.type_id());

    std.debug.assert(std.mem.eql(usize, x.sizes(), y.sizes()));

    const z = graph.tensor(.{ .class = .hid, .dtype = x.type_tag(), .sizes = x.sizes() });

    core.kernels.hadamard[z.type_id()](
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

pub fn reverse(
    args: []const OpDatum,
    type_id: usize,
) void {

    core.enable_gradient(args[0].tensor);
    core.enable_gradient(args[1].tensor);

    const dispatch = core.kernels.hadamard_reverse[type_id];

    dispatch(
        args[2].tensor.grad_ptr(),
        args[1].tensor.data_ptr(),
        args[0].tensor.grad_ptr(),
        args[0].tensor.len(),
        args[0].tensor.stream(),
    );
    dispatch(
        args[2].tensor.grad_ptr(),
        args[0].tensor.data_ptr(),
        args[1].tensor.grad_ptr(),
        args[1].tensor.len(),
        args[1].tensor.stream(),
    );
}

const dilate = @import("dilate.zig");
const addition = @import("addition.zig");

pub fn derive(args: []const OpDatum, wrt: Tensor) ?OpDatum {

    // common optimization for power rule
    if (args[0].tensor.same(wrt) and args[1].tensor.same(wrt)) {
        return OpDatum{ .tensor = dilate.forward_impl(wrt.ptr, args[0].tensor, 2.0) };
    }

    const dx: OpDatum = core.derive(args[0].tensor, wrt) orelse {
        
        const dy = core.derive(args[1].tensor, wrt) orelse return null;

        if (dy == .scalar) { // c * f'(x)

            if (is_zero(dy.scalar)) // f'(x): 0 -> 0*c = 0
                return dy;

            if (is_one(dy.scalar)) // f'(x): 1 -> 1*c = c
                return args[0];
            
            return OpDatum{ .tensor = dilate.forward_impl(wrt.ptr, args[0].tensor, dy.scalar) };
        }
        return OpDatum{ .tensor = forward_impl(wrt.ptr, args[0].tensor, dy.tensor) };  
    };

    const dy: OpDatum = core.derive(args[1].tensor, wrt) orelse {

        if (dx == .scalar) { // c * f'(x)

            if (is_zero(dx.scalar)) // f'(x): 0 -> 0*c = 0
                return dx;

            if (is_one(dx.scalar)) // f'(x): 1 -> 1*c = c
                return args[1];

            return OpDatum{ .tensor = dilate.forward_impl(wrt.ptr, args[1].tensor, dx.scalar) };
        }
        return OpDatum{ .tensor = forward_impl(wrt.ptr, args[1].tensor, dx.tensor) };
    };


    if (dx == .scalar and dy == .scalar) {

        if (is_zero(dx.scalar) and is_zero(dy.scalar))
            return null;
        
        if (is_zero(dx.scalar)) // only keep rhs
            return OpDatum{ .tensor = dilate.forward_impl(wrt.ptr, args[0].tensor, dy.scalar) };

        if (is_zero(dy.scalar)) // only keep lhs
            return OpDatum{ .tensor = dilate.forward_impl(wrt.ptr, args[1].tensor, dx.scalar) };

        if (is_one(dx.scalar) and is_one(dy.scalar)) // x': 1, y': 1 -> 1y + x1 = x + y
            return OpDatum{ .tensor = addition.forward_impl(wrt.ptr, args[0].tensor, args[1].tensor) };

        if (is_one(dx.scalar)) { // x': 1 -> xy' + 1y = xy' + y
            const u = dilate.forward_impl(wrt.ptr, args[0].tensor, dy.scalar);
            return OpDatum{ .tensor = addition.forward_impl(wrt.ptr, u, args[1].tensor) };
        }
        if (is_one(dy.scalar)) { // y': 1 -> x1 + x'y = x + x'y
            const u = dilate.forward_impl(wrt.ptr, args[1].tensor, dx.scalar);
            return OpDatum{ .tensor = addition.forward_impl(wrt.ptr, u, args[0].tensor) };
        }
        // neither dx or dy are equal to one or zero
        const t = dilate.forward_impl(wrt.ptr, args[0].tensor, dy.scalar);
        const u = dilate.forward_impl(wrt.ptr, args[1].tensor, dx.scalar);
        return OpDatum{ .tensor = addition.forward_impl(wrt.ptr, t, u) };
    }

    if (dx == .scalar) {
        if (is_zero(dx.scalar)) // x': 0 -> xy' + y0 = xy'
            return OpDatum{ .tensor = forward_impl(wrt.ptr, args[0].tensor, dy.tensor) };
        
        if (is_one(dx.scalar)) { // x': 1 -> xy' + y1 = xy' + y
            const t = forward_impl(wrt.ptr, args[0].tensor, dy.tensor);
            return OpDatum{ .tensor = addition.forward_impl(wrt.ptr, t, args[1].tensor) };
        }
        const t = forward_impl(wrt.ptr, args[0].tensor, dy.tensor);
        const u = dilate.forward_impl(wrt.ptr, args[1].tensor, dx.scalar);
        return OpDatum{ .tensor = addition.forward_impl(wrt.ptr, t, u) };
    }

    if (dy == .scalar) {
        if (is_zero(dy.scalar)) // y': 0 -> x0 + yx' = yx'
            return OpDatum{ .tensor = forward_impl(wrt.ptr, args[1].tensor, dx.tensor) };
        
        if (is_one(dy.scalar)) { // y': 1 -> x1 + yx' = x + yx'
            const t = forward_impl(wrt.ptr, args[1].tensor, dx.tensor);
            return OpDatum{ .tensor = addition.forward_impl(wrt.ptr, args[0].tensor, t) };
        }
        const t = forward_impl(wrt.ptr, args[1].tensor, dx.tensor);
        const u = dilate.forward_impl(wrt.ptr, args[0].tensor, dy.scalar);
        return OpDatum{ .tensor = addition.forward_impl(wrt.ptr, t, u) };
    }

    const t = forward_impl(wrt.ptr, args[0].tensor, dy.tensor);
    const u = forward_impl(wrt.ptr, args[1].tensor, dx.tensor);
    return OpDatum{ .tensor = addition.forward_impl(wrt.ptr, t, u) };
}
