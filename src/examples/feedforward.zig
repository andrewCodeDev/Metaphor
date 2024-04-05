const mp = @import("metaphor");
const EU = @import("example_utils.zig");
const std = @import("std");

pub fn FeedForward(comptime Tag: mp.scalar.Tag) type {
    return struct {
        const Self = @This();
        const T = mp.scalar.Tag.asType(Tag);

        W: mp.types.LeafTensor(T, .wgt),
        b: mp.types.LeafTensor(T, .wgt),
        y: mp.types.NodeTensor(T) = undefined,
        alpha: f32,

        pub fn init(G: *mp.Graph, m: usize, n: usize) Self {
            return .{
                .W = G.tensor(.wgt, Tag, mp.Rank(2){ m, n }),
                .b = G.tensor(.wgt, Tag, mp.Rank(1){m}),
                .alpha = 1.0 / @as(f16, @floatFromInt(n)),
            };
        }

        pub fn randomize(self: *Self) void {
            mp.mem.randomize(self.W, .gauss);
            mp.mem.randomize(self.b, .gauss);
        }

        pub fn forward(self: *Self, x: anytype) mp.types.NodeTensor(T) {
            self.y = mp.ops.selu(mp.ops.linearScaled(self.W, x, self.alpha, self.b, "ij,j->i"));
            self.y.detach();
            return self.y;
        }

        pub fn reverse(self: *Self, cleanup: bool) void {
            self.y.reverse(if (cleanup) .free else .keep);
        }
    };
}

pub fn NeuralNet(comptime Tag: mp.scalar.Tag, comptime layers: usize) type {
    if (comptime layers == 0) {
        @compileError("NeuralNet needs at least 1 layer.");
    }

    return struct {
        const Self = @This();
        const T = mp.scalar.Tag.asType(Tag);

        head: FeedForward(Tag),
        body: [layers]FeedForward(Tag),
        tail: FeedForward(Tag),
        cleanup: bool,

        pub fn init(G: *mp.Graph, m: usize, n: usize, cleanup: bool) Self {
            var body: [layers]FeedForward(Tag) = undefined;
            for (0..layers) |i| {
                body[i] = FeedForward(Tag).init(G, m, m);
            }
            return .{
                .head = FeedForward(Tag).init(G, m, n),
                .body = body,
                .tail = FeedForward(Tag).init(G, n, m),
                .cleanup = cleanup,
            };
        }

        pub fn forward(self: *Self, x: mp.types.LeafTensor(T, .inp)) mp.types.NodeTensor(T) {
            var h = self.head.forward(x);

            for (0..layers) |i| {
                h = self.body[i].forward(h);
            }

            return self.tail.forward(h);
        }

        pub fn reverse(self: *Self) void {
            self.tail.reverse(self.cleanup);

            var i: usize = layers;
            while (i > 0) {
                i -= 1;
                self.body[i].reverse(self.cleanup);
            }

            self.head.reverse(self.cleanup);
        }

        pub fn randomize(self: *Self) void {
            self.head.randomize();
            self.tail.randomize();
            for (0..layers) |i| self.body[i].randomize();
        }
    };
}

pub fn main() !void {
    mp.device.init(0);

    const stream = mp.stream.init();
    defer mp.stream.deinit(stream);

    var sgd = mp.optm.SGD.init(.{ .rate = 0.1, .clip = .{
        .lower = -2.0,
        .upper = 2.0,
    } });

    const G = mp.Graph.init(.{ .stream = stream, .mode = .train });
    defer G.deinit();

    const m: usize = 128;
    const n: usize = 32;

    /////////////////////////////////////////////////////
    // feed forward network...

    var net = NeuralNet(.r32, 3).init(G, m, n, true);
    const x = G.tensor(.inp, .r32, mp.Rank(1){n});
    const t = 16;

    mp.mem.randomize(x, .gauss);
    net.randomize();

    var score: f32 = 0.0;

    for (0..100) |_| {
        const y = net.forward(x);

        mp.loss.cce(y, t, .{
            .grads = true,
            .score = &score,
        });

        net.reverse();

        sgd.update(G);

        G.reset(.node, .all);
        G.reset(.leaf, .grd);

        std.log.info("score: {d:.4}", .{score});
    }

    ////////////////////////////////////////////
    mp.device.check();

    std.log.info("Feedforward: SUCCESS", .{});
}
