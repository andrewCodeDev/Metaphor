const mp = @import("metaphor");
const EU = @import("example_utils.zig");
const std = @import("std");

const BytesList = std.ArrayList([]const u8);
const BytesMap = std.StringHashMap(std.ArrayList([]const u8));
const MapKeys = std.ArrayList([]const u8);

const ExampleClass = enum { T, F };

pub fn getExampleClass(name: []const u8) ExampleClass {
    return if (std.mem.indexOfScalar(u8, name, 'T') != null)
        ExampleClass.T else ExampleClass.F;
}

pub fn addExample(
    allocator: std.mem.Allocator,
    T_map: *BytesMap, 
    F_map: *BytesMap, 
    map_keys: *MapKeys,
    key: []const u8,
    bytes: []const u8
) !void {

    const end = std.mem.indexOfAny(u8, key, "TF") orelse {
        std.log.err("Bad file name: {s}", .{ key });
        @panic("Bad file name.");
    };

    const new_key = try allocator.dupe(u8, key[0..end]);

    // we need to keep the maps in sync
    const T_res = try T_map.getOrPut(new_key);
    const F_res = try F_map.getOrPut(new_key);

    if (!T_res.found_existing)
        T_res.value_ptr.* = try BytesList.initCapacity(allocator, 20);

    if (!F_res.found_existing)
        F_res.value_ptr.* = try BytesList.initCapacity(allocator, 20);

    switch (getExampleClass(key)) {
        .T => try T_res.value_ptr.append(bytes),
        .F => try F_res.value_ptr.append(bytes),
    }
    
    // only append map key if we haven't seen it yet
    if (!(T_res.found_existing or F_res.found_existing)) {
        try map_keys.append(new_key);
    }
}

pub fn getExample(
    T_map: *BytesMap, 
    F_map: *BytesMap, 
    map_keys: []const []const u8,
    rand: std.Random,
) struct { 
    q_val: []const u8, 
    k_val: []const u8,
    class: ExampleClass,
    key: []const u8,
} {

    const class = if (rand.boolean())
        ExampleClass.T else ExampleClass.F;

    const key = blk: {
        const n = map_keys.len;
        const i = rand.uintLessThan(usize, n);
        break :blk map_keys[i];
    };

    // we may use this again
    const T_items = T_map.get(key).?.items;
    
    // the model always needs a true example
    // to say if there is fraudulent example
    const k_val = blk: {
        const i = rand.uintLessThan(usize, T_items.len);
        break :blk T_items[i];
    };

    const q_val = switch (class) {
        .T => blk: {
            const i = rand.uintLessThan(usize, T_items.len);
            break :blk T_items[i];
        },
        .F => blk: {
            const F_items = F_map.get(key).?.items;
            const i = rand.uintLessThan(usize, F_items.len);
            break :blk F_items[i];
        }
    };

    return .{ .q_val = q_val, .k_val = k_val, .class = class, .key = key };
}

pub fn populateExamples(
    allocator: std.mem.Allocator,
    T_ptr: *BytesMap,
    F_ptr: *BytesMap,
    keys_ptr: *MapKeys,
) !void {
    var dir: std.fs.Dir = try std.fs.cwd().openDir("TrainingData", .{ 
        .access_sub_paths = false, 
        .iterate = true, 
        .no_follow = true,
    });

    defer dir.close();
    
    var fullpath = std.BoundedArray(u8, 32).init(0) catch unreachable;

    var itr = dir.iterate();

    while (try itr.next()) |path| {

        try std.fmt.format(fullpath.writer(), "TrainingData/{s}", .{ path.name });

        const bytes = try std.fs.cwd().readFileAlloc(allocator, fullpath.slice(), std.math.maxInt(usize));
        
        try addExample(allocator, T_ptr, F_ptr, keys_ptr, path.name, bytes);

        fullpath.resize(0) catch unreachable;
    }
    // Trim bad examples
    var removed: usize = 0;

    var i: usize = 0;

    while (i < keys_ptr.items.len) {
        var remove: u1 = 0;
        remove |= @intFromBool(T_ptr.get(keys_ptr.items[i]).?.items.len == 0);
        remove |= @intFromBool(F_ptr.get(keys_ptr.items[i]).?.items.len == 0);

        if (remove != 0) {
            _ = T_ptr.remove(keys_ptr.items[i]);
            _ = F_ptr.remove(keys_ptr.items[i]);
            _ = keys_ptr.swapRemove(i);
            removed += 1;
            continue;
        }

        i += 1;    
    }
    std.log.info("removed - {}", .{ removed });
}

const Attention = struct {
    Q: mp.types.LeafTensor(mp.scalar.r32, .wgt),
    K: mp.types.LeafTensor(mp.scalar.r32, .wgt),
    V: mp.types.LeafTensor(mp.scalar.r32, .wgt),
    W1: mp.types.LeafTensor(mp.scalar.r32, .wgt),
    b1: mp.types.LeafTensor(mp.scalar.r32, .wgt),
    W2: mp.types.LeafTensor(mp.scalar.r32, .wgt),
    b2: mp.types.LeafTensor(mp.scalar.r32, .wgt),
    alpha: f32,
    Y: mp.types.NodeTensor(mp.scalar.r32) = undefined,

    pub fn init(G: *mp.Graph, m: usize, n: usize) Attention {
        return .{
            .Q = G.tensor(.wgt, .r32, mp.Rank(2){ n, m }),
            .K = G.tensor(.wgt, .r32, mp.Rank(2){ n, m }),
            .V = G.tensor(.wgt, .r32, mp.Rank(2){ n, m }),
            .W1 = G.tensor(.wgt, .r32, mp.Rank(2){ 4 * n, n }),
            .b1 = G.tensor(.wgt, .r32, mp.Rank(2){ 4 * n, n }),
            .W2 = G.tensor(.wgt, .r32, mp.Rank(2){ m, 4 * n }),
            .b2 = G.tensor(.wgt, .r32, mp.Rank(2){ m, n }),
            .alpha = 1.0 / @as(f32, @floatFromInt(n)),
        };
    }

    pub fn forward(self: *Attention, q: anytype, k: anytype) @TypeOf(self.Y) {

        const QX = mp.ops.innerProduct(self.Q, q, "ij,jk->ik");
        const KX = mp.ops.innerProduct(self.K, k, "ij,jk->ik");
        const VX = mp.ops.innerProduct(self.V, q, "ij,jk->ik");

        // calculate overlap
        const QK = mp.ops.innerProduct(QX, KX, "ij,kj->ik");
        const SM = mp.ops.softmax(QK, "ij|j");
        const SV = mp.ops.innerProductScaled(SM, VX, self.alpha, "ij,jk->ik");
        
        const Z1 = mp.ops.selu(mp.ops.linear(self.W1, SV, self.b1, "ij,jk->ik"));
        const Z2 = mp.ops.linear(self.W2, Z1, self.b2, "ij,jk->ik");
        self.Y = mp.ops.add(mp.ops.norm.minmax(Z2, "ij|j"), q);

        self.Y.detach();

        return self.Y;
    }

    pub fn reverse(self: *Attention) void {
        return self.Y.reverse(.free);
    }

    pub fn randomize(self: *Attention) void {
        mp.mem.randomize(self.Q, .gauss);
        mp.mem.randomize(self.K, .gauss);
        mp.mem.randomize(self.V, .gauss);
        mp.mem.randomize(self.W1, .gauss);
        mp.mem.randomize(self.b1, .gauss);
        mp.mem.randomize(self.W2, .gauss);
        mp.mem.randomize(self.b2, .gauss);
    }
};

pub fn main() !void {

    ///////////////////////////////////////
    // Initial graph and examples setup //

    mp.device.init(0);

    const stream = mp.stream.init();
    defer mp.stream.deinit(stream);

    const G = mp.Graph.init(.{ .stream = stream, .mode = .train });
    defer G.deinit();

    const x = G.tensor(.wgt, .r32, mp.Rank(2){ 6, 6 });
    const k = G.tensor(.wgt, .r32, mp.Rank(2){ 2, 2 });

    mp.mem.fill(x, 1.00);
    mp.mem.fill(k, 1.00);

    const y = mp.ops.convolution(x, k, .{ .dims = 2, .channels = 1, .stride = 3 }); 

    y.reverse(.keep);

    try EU.copyAndPrintMatrix("y", y.values(), y.sizes()[0], y.sizes()[0], stream);
    try EU.copyAndPrintMatrix("x grads", x.grads().?, x.sizes()[0], x.sizes()[1], stream);
    try EU.copyAndPrintMatrix("k grads", k.grads().?, k.sizes()[0], k.sizes()[1], stream);

//    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
//    defer arena.deinit();
//
//    var pcg = std.Random.Pcg.init(42);
//
//    var mtm = mp.optm.Momentum.init(.{ 
//        .rate = 0.001, .alpha = 0.99, .clip = .{ .lower = -1.0, .upper = 1.0 } 
//    });
//
//    //defer mtm.deinit(stream);
//    //
//    var T_map: BytesMap = BytesMap.init(arena.allocator());
//    var F_map: BytesMap = BytesMap.init(arena.allocator());
//    var map_keys: MapKeys = MapKeys.init(arena.allocator());
//
//    ///////////////////////////////////////
//    // populate our training examples /////
//
//    try populateExamples(arena.allocator(), &T_map, &F_map, &map_keys);
//
//    const split: usize = blk: {
//        const len: f32 = mp.scalar.as(f32, map_keys.items.len);  
//        const out: f32 = len * 0.1;
//        break :blk @intFromFloat(out);
//    };
//
//    const train_split = map_keys.items[0..map_keys.items.len - split];
//    const test_split = map_keys.items[map_keys.items.len - split..];
//
//    ///////////////////////////////////////
//    // create weight and input tensors ////
//
//    const m: usize = 64;
//    const n: usize = 128;
//
//    const x1 = G.tensor(.inp, .r32, mp.Rank(2){ m, n });
//    const x2 = G.tensor(.inp, .r32, mp.Rank(2){ m, n });
//
//    const U1 = G.tensor(.wgt, .r32, mp.Rank(2){ 2, m });
//    const U2 = G.tensor(.wgt, .r32, mp.Rank(2){ 2, n });
//
//    var attn_a = Attention.init(G, m, n);
//    var attn_b = Attention.init(G, m, n);
//    var attn_c = Attention.init(G, m, n);
//
//    attn_a.randomize();
//    attn_b.randomize();
//    attn_c.randomize();
//
//    mp.mem.randomize(U1, .gauss);
//    mp.mem.randomize(U2, .gauss);
//
//    const alpha: f32 = 1.0 / @as(f32, @floatFromInt(n));
//
//    var score: f32 = 0.0;
//
//    for (0..200_000) |epoch| {
//
//        const example = getExample(&T_map, &F_map, train_split, pcg.random());
//        mp.mem.copyToDevice(example.q_val, x1.raw_values().bytes(), stream);
//        mp.mem.copyToDevice(example.k_val, x2.raw_values().bytes(), stream);
//
//        const t: mp.types.Key = if (example.class == .T) 
//            @as(mp.types.Key, 0) else @as(mp.types.Key, 1);
//
//        const z1 = attn_a.forward(x1, x2);
//        const z2 = attn_b.forward(z1, x2);
//        const z3 = attn_c.forward(z2, x2);
//        
//        const p1 = mp.ops.innerProductScaled(U1, z3, alpha, "ij,jk->ik");
//        const p2 = mp.ops.reduce(mp.ops.hadamard(U2, p1), "ij->i");
//                
//        mp.loss.cce(p2, t, .{
//            .grads = true,
//            .score = &score
//        });
//
//        if (epoch % 1000 == 0)
//            std.log.info("score - {d:.5}, epoch - {}", .{ score, epoch });
//
//        p2.reverse(.free);
//        attn_c.reverse();
//        attn_b.reverse();
//        attn_a.reverse();
//
//        mtm.update(G);
//
//        G.reset(.node, .all);
//        G.reset(.leaf, .grd);
//    }
//
//    mp.device.check();
//
//    //////////////////////////
//    //// TEST BLOCK //////////
//
//    G.mode = .eval;
//
//    std.log.info("Testing: ", .{});
//
//    for (0..100) |_| {
//
//        const example = getExample(&T_map, &F_map, test_split, pcg.random());
//        mp.mem.copyToDevice(example.q_val, x1.raw_values().bytes(), stream);
//        mp.mem.copyToDevice(example.k_val, x2.raw_values().bytes(), stream);
//
//        const t: mp.types.Key = if (example.class == .T) 
//            @as(mp.types.Key, 0) else @as(mp.types.Key, 1);
//
//        const z1 = attn_a.forward(x1, x2);
//        const z2 = attn_b.forward(z1, x2);
//        const z3 = attn_b.forward(z2, x2);
//        
//        const p1 = mp.ops.innerProductScaled(U1, z3, alpha, "ij,jk->ik");
//        const p2 = mp.ops.reduce(mp.ops.hadamard(U2, p1), "ij->i");
//                
//        mp.loss.cce(p2, t, .{
//            .grads = false,
//            .score = &score
//        });
//
//        std.debug.print("\n-------------\n", .{});
//
//        std.log.info("score - {d:.5}, {}", .{ score, t });
//
//        try EU.copyAndPrintMatrix("Out", p2.values(), 1, 2, stream);
//
//        G.reset(.node, .all);
//    }
//    //////////////////////////////////////////////
//
//    mp.device.check();
//
//    std.log.info("Experimental: SUCCESS", .{});
}
