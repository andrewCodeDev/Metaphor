
 /////////////////////////////////////////////
/////////////////////////////////////////////

const SigmoidReverse = struct {
    pub fn reverseArg0(grads: anytype, _: anytype, Z: anytype) void {
        const z = Z.values();
        const g = Z.grads().?;
        for (0..z.len) |i| {
            grads[i] += g[i] * (z[i] * (1.0 - z[i]));
        }
    }
};

pub fn sigmoid(X: anytype) !Contract(
        isGraphTensor(@TypeOf(X)),
    GraphTensor(@TypeOf(X).DataType)) {

        const G = X.graph_ptr;
        const x = X.values();
        const z = try G.tensor_allocator.alloc(@TypeOf(X).DataType, x.len);

        for (0..z.len) |i| {
            z[i] = 1.0 / (1.0 + std.math.exp(-x[i]));
        }

        var Z = try G.tensorFromComponents("", .hid, z, X.sizes(), X.strides());

        return try G.appendNode(SigmoidReverse, .{ X }, &Z);
    }

 /////////////////////////////////////////////
/////////////////////////////////////////////

fn ReluReverse(comptime rate: anytype) type {
    return struct {
        const Rate = rate;
        pub fn reverseArg0(grads: anytype, X: anytype, Z: anytype) void {
            const x = X.values();
            const g = Z.grads().?;
            for (0..x.len) |i| {
                grads[i] += g[i] * (if (x[i] < 0) Rate else 1);
            }
        }
    };
}

pub fn relu(
    X: anytype, 
    comptime rate: @TypeOf(X).DataType) !Contract(
        isFloat(@TypeOf(rate)) and
        isGraphTensor(@TypeOf(X)), 
    GraphTensor(@TypeOf(X).DataType)) {

        const G = X.graph_ptr;
        const x = X.values();
        const z = try G.tensor_allocator.alloc(@TypeOf(X).DataType, x.len);

        for (0..z.len) |i| {
            z[i] = x[i] * (if (x[i] < 0) rate else 1);
        }

        var Z = try G.tensorFromComponents("", .hid, z, X.sizes(), X.strides());

        return try G.appendNode(ReluReverse(rate), .{ X }, &Z);
    }

 /////////////////////////////////////////////
/////////////////////////////////////////////

