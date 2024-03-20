const std = @import("std");
const mem = std.mem;
const ArrayList = std.ArrayList;
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

const UT = @import("utility.zig");
const SC = @import("scalar.zig");
const TC = @import("tensor_components.zig");
const DU = @import("device_utils.zig");

const Optm = @import("optimizer.zig");
const Optimizer = Optm.Optimizer;
const LaneAllocator = @import("lane_allocator.zig").LaneAllocator;
const Stream = DU.Stream;

const isReal = SC.isReal;
const isFloat = SC.isFloat;
const isInteger = SC.isInteger;
const isComplex = SC.isComplex;
const isScalar = SC.isScalar;
const c16 = SC.c16;
const c32 = SC.c32;
const c64 = SC.c64;
const ScalarTag = SC.ScalarTag;

// default optimizer if null is passed via graph config
const null_optimizer = Optm.NullOptimizer.optimizer();

const Contract = UT.Contract;
const Returns = UT.Returns;
const Child = UT.Child;

const kernel_fill = @import("kernel_overloads.zig").kernel_fill;

const isFunction = UT.isFunction;
const isPointer = UT.isPointer;
const isSlice = UT.isSlice;
const isArray = UT.isArray;
const isStruct = UT.isStruct;
const isTuple = UT.isTuple;

// implementation declarations
const getSlice = TC.getSlice;
const SliceUnion = TC.SliceUnion;
const IndexType = TC.IndexType;
const SizeType = TC.SizeType;
const Strides = TC.Strides;
const Sizes = TC.Sizes;

pub fn isGraphTensor(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .Struct => if (!@hasDecl(T, "DataType") or !@hasDecl(T, "Class")) false else (T == GraphTensor(T.DataType, T.Class)),
        else => false,
    };
}

// TODO: move this function?
fn fillSlice(
    comptime T: type,
    x_slice: []T,
    value: T,
    stream: Stream,
) void {
    kernel_fill.call(.{ stream.context, x_slice.ptr, value, x_slice.len });
    DU.synchronizeStream(stream);
}

pub fn fill(X: anytype, value: anytype) Contract(isGraphTensor(@TypeOf(X)), Returns(void)) {
    const T = Child(@TypeOf(X));

    fillSlice(T, X.values(), SC.asScalar(T, value), X.ptr.stream);
}

pub const TensorClass = enum {
    inp,
    wgt,
    hid,
};

pub fn LeafTensor(comptime T: type, comptime class: TensorClass) type {
    if (comptime class == .hid) {
        @compileError("Cannot instantiate a leaf with class type of hidden.");
    }

    return struct {
        const Self = @This();
        pub const DataType = T;
        pub const Class = class;

        ptr: *Graph,
        idx: IndexType,

        pub fn values(self: Self) []DataType {
            return getSlice(DataType, self.raw_values());
        }

        pub fn raw_values(self: Self) SliceUnion {
            return self.ptr.leaves.values.items[self.idx];
        }

        pub fn sizes(self: Self) Sizes {
            return self.ptr.leaves.sizes.items[self.idx];
        }

        pub fn strides(self: Self) Strides {
            return self.ptr.leaves.strides.items[self.idx];
        }

        pub fn raw_grads(self: Self) ?SliceUnion {
            return self.ptr.leaves.grads.items[self.idx];
        }

        pub fn grads(self: Self) ?[]DataType {
            if (self.raw_grads()) |grd| {
                return getSlice(DataType, grd);
            }
            return null;
        }

        pub fn len(self: Self) SizeType {
            return self.values().len;
        }

        pub fn stream(self: Self) Stream {
            return self.ptr.leaves.streams.items[self.idx];
        }

        pub fn free(self: Self) void {
            self.ptr.freeTensor(DataType, Class, self.idx);
        }

        fn adjustDependencies(self: Self, comptime direction: i8) i8 {
            return self.ptr.adjustDependencies(Class, self.idx, direction);
        }
    };
}

pub fn NodeTensor(comptime T: type) type {
    return struct {
        const Self = @This();

        pub const DataType = switch (@typeInfo(T)) {
            .Struct => T.DataType,
            else => T,
        };
        pub const Class = TensorClass.hid;

        ptr: *Graph,
        idx: IndexType,

        pub fn values(self: Self) []DataType {
            return getSlice(DataType, self.raw_values());
        }

        pub fn raw_values(self: Self) SliceUnion {
            return self.ptr.nodes.values.items[self.idx];
        }

        pub fn sizes(self: Self) Sizes {
            return self.ptr.nodes.sizes.items[self.idx];
        }

        pub fn strides(self: Self) Strides {
            return self.ptr.nodes.strides.items[self.idx];
        }

        pub fn raw_grads(self: Self) ?SliceUnion {
            return self.ptr.nodes.grads.items[self.idx];
        }

        pub fn grads(self: Self) ?[]DataType {
            return if (self.raw_grads()) |grd| getSlice(DataType, grd) else null;
        }

        pub fn len(self: Self) SizeType {
            return self.values().len;
        }

        pub fn stream(self: Self) Stream {
            return self.ptr.nodes.streams.items[self.idx];
        }

        pub fn attach(self: Self) void {
            self.ptr.setAttachment(self.idx, true);
        }

        pub fn detach(self: Self) void {
            self.ptr.setAttachment(self.idx, false);
        }

        pub fn attached(self: Self) bool {
            return self.ptr.attached(self.idx);
        }

        pub fn free(self: Self) void {
            self.ptr.freeTensor(DataType, Class, self.idx);
        }

        pub fn reverse(self: Self, cleanup: ReverseCleanup) void {
            if (self.grads() == null) {
                // if a loss hasn't been applied, this will be null
                const grd = enableGradient(self.ptr, DataType, .hid, self.idx);
                // derivative with respect to self is 1
                fillSlice(DataType, grd, SC.asScalar(DataType, 1.0), self.ptr.stream);
            }

            // call graph reverse
            self.ptr.reverse(self.idx, cleanup);
        }
        fn adjustDependencies(self: Self, comptime direction: i8) i8 {
            return self.ptr.adjustDependencies(.hid, self.idx, direction);
        }
    };
}

fn GraphTensor(comptime data_type: type, comptime class: TensorClass) type {
    return if (comptime class == .hid) NodeTensor(data_type) else LeafTensor(data_type, class);
}

const Mode = enum {
    train,
    eval,
};

pub const GraphConfig = struct {
    optimizer: ?Optimizer = null,
    stream: Stream,
    mode: Mode,
};

pub const ReverseCleanup = enum {
    keep,
    free,
};

pub const Graph = struct {
    const Leaves = struct {
        values: ArrayList(SliceUnion),
        grads: ArrayList(?SliceUnion),
        sizes: ArrayList(Sizes),
        strides: ArrayList(Strides),
        dependencies: ArrayList(i8),
        streams: ArrayList(Stream),

        pub fn init(allocator: std.mem.Allocator, block_size: usize) !Leaves {
            return @This(){
                .values = try ArrayList(SliceUnion).initCapacity(allocator, block_size),
                .grads = try ArrayList(?SliceUnion).initCapacity(allocator, block_size),
                .sizes = try ArrayList(Sizes).initCapacity(allocator, block_size),
                .strides = try ArrayList(Strides).initCapacity(allocator, block_size),
                .dependencies = try ArrayList(i8).initCapacity(allocator, block_size),
                .streams = try ArrayList(Stream).initCapacity(allocator, block_size),
            };
        }
    };

    const Nodes = struct {
        callbacks: ArrayList(Closure),
        values: ArrayList(SliceUnion),
        grads: ArrayList(?SliceUnion),
        sizes: ArrayList(Sizes),
        strides: ArrayList(Strides),
        dependencies: ArrayList(i8),
        attached: ArrayList(bool),
        streams: ArrayList(Stream),

        pub fn init(allocator: std.mem.Allocator, block_size: usize) !Nodes {
            return .{
                .callbacks = try ArrayList(Closure).initCapacity(allocator, block_size),
                .values = try ArrayList(SliceUnion).initCapacity(allocator, block_size),
                .grads = try ArrayList(?SliceUnion).initCapacity(allocator, block_size),
                .sizes = try ArrayList(Sizes).initCapacity(allocator, block_size),
                .strides = try ArrayList(Strides).initCapacity(allocator, block_size),
                .dependencies = try ArrayList(i8).initCapacity(allocator, block_size),
                .attached = try ArrayList(bool).initCapacity(allocator, block_size),
                .streams = try ArrayList(Stream).initCapacity(allocator, block_size),
            };
        }
    };

    node_max: usize,
    leaf_max: usize,

    leaf_arena: std.heap.ArenaAllocator,
    node_arena: std.heap.ArenaAllocator,

    tensor_allocator: LaneAllocator,

    // optimizer interface to update weights
    optimizer: Optimizer,

    // tensor stuff...
    leaves: Leaves,
    nodes: Nodes,

    mode: Mode,

    stream: Stream,

    pub fn id(self: *const Graph) usize {
        return @intFromPtr(self);
    }

    // this function ensures the correct allocator
    // and size are passed to the designated block
    fn initLeaves(self: *Graph) Leaves {
        return Leaves.init(self.leaf_arena.allocator(), self.leaf_max) catch @panic("Graph.initLeaves: Out of Memory");
    }

    // this function ensures the correct allocator
    // and size are passed to the designated block
    fn initNodes(self: *Graph) Nodes {
        return Nodes.init(self.node_arena.allocator(), self.node_max) catch @panic("Graph.initNodes: Out of Memory");
    }

    pub fn init(config: GraphConfig) *Graph {
        const self = std.heap.c_allocator.create(Graph) catch @panic("Graph.init: Out of Memory");

        self.optimizer = config.optimizer orelse null_optimizer;

        self.stream = config.stream;

        self.leaf_max = 0;
        self.node_max = 0;

        self.leaf_arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
        self.node_arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);

        self.leaves = self.initLeaves();
        self.nodes = self.initNodes();

        self.tensor_allocator.setup();

        self.mode = config.mode;

        return self;
    }

    pub fn deinit(self: *Graph) void {
        // hand our memory back to the allocator
        self.reset(.node, .all);
        self.reset(.leaf, .all);

        // free our memory from the allocator
        self.tensor_allocator.deinit(self.stream);

        self.node_arena.deinit();
        self.leaf_arena.deinit();

        // this must always be last
        std.heap.c_allocator.destroy(self);
    }

    const ResetGroup = enum {
        node,
        leaf,
    };

    const ResetComponent = enum {
        all,
        grd,
    };

    // this function helps defragment the arenas and
    // preps the graph - called by reverse conditionally
    pub fn reset(self: *Graph, group: ResetGroup, component: ResetComponent) void {
        if (group == .leaf) {
            if (component == .grd) {
                for (self.leaves.grads.items, 0..) |opt, i| {
                    if (opt) |raw| self.freeGradientRaw(raw, .inp, i);
                }
            }
            if (component == .all) {
                // frees both values and grads
                for (self.leaves.values.items, 0..) |raw, i| {
                    if (0 < raw.len()) self.freeTensorRaw(raw, .inp, i);
                }
                // keep the max number of leaves created
                self.leaf_max = @max(self.leaf_max, self.leaves.values.items.len);

                // drop all of our values out of play
                _ = self.leaf_arena.reset(.retain_capacity);

                // reinstantiate the array lists
                self.leaves = self.initLeaves();
            }
        } else {
            if (component == .grd) {
                // frees both values and grads
                for (self.nodes.grads.items, 0..) |opt, i| {
                    if (opt) |raw| self.freeGradientRaw(raw, .hid, i);
                }
            }
            if (component == .all) {
                for (self.nodes.values.items, 0..) |raw, i| {
                    if (0 < raw.len()) self.freeTensorRaw(raw, .hid, i);
                }
                // keep the max number of leaves created
                self.node_max = @max(self.node_max, self.nodes.values.items.len);

                // drop all of our values out of play
                _ = self.node_arena.reset(.retain_capacity);

                // reinstantiate the array lists
                self.nodes = self.initNodes();
            }
        }
    }

    // load precache values for the tensor allocator to use
    pub fn precache(self: *Graph, comptime T: type, size: usize, count: usize) void {
        self.tensor_allocator.precache(T, size, count, self.stream);
    }

    fn tensorFromComponents(
        self: *Graph,
        comptime class: TensorClass,
        values: anytype,
        sizes: Sizes,
        strides: Sizes,
    ) !GraphTensor(Child(@TypeOf(values)), class) {
        assert(0 < sizes.len);
        assert(0 < strides.len);

        if (comptime class == .hid) {
            try self.nodes.values.append(SliceUnion.init(values));
            try self.nodes.sizes.append(sizes);
            try self.nodes.strides.append(strides);
            try self.nodes.grads.append(null);
            try self.nodes.dependencies.append(0);
            try self.nodes.attached.append(true);
            try self.nodes.streams.append(self.stream);
        } else {
            try self.leaves.values.append(SliceUnion.init(values));
            try self.leaves.sizes.append(sizes);
            try self.leaves.strides.append(strides);
            try self.leaves.grads.append(null);
            try self.leaves.dependencies.append(0);
            try self.leaves.streams.append(self.stream);
        }

        const idx = if (class == .hid) self.nodes.values.items.len - 1 else self.leaves.values.items.len - 1;

        return GraphTensor(Child(@TypeOf(values)), class){
            .ptr = self,
            .idx = idx,
        };
    }

    inline fn setAttachment(self: *Graph, idx: IndexType, state: bool) void {
        self.nodes.attached.items[idx] = state;
    }

    inline fn attached(self: *Graph, idx: IndexType) bool {
        return self.nodes.attached.items[idx];
    }

    // Users can only provide inp or wgt,
    // so for the sake of the UI, the enum
    // here only exposes the two valid options

    fn userTensor(comptime class: anytype) TensorClass {
        return if (comptime class == .inp) TensorClass.inp else TensorClass.wgt;
    }

    fn constructTensor(
        self: *Graph,
        comptime class: TensorClass,
        dimensions: Sizes, // fixed-length array
        comptime data_type: type,
        allocator: std.mem.Allocator,
    ) GraphTensor(data_type, class) {
        assert(0 < dimensions.len);

        const N = blk: {
            var n: SizeType = 1;
            for (dimensions) |s| {
                n *= s;
            }
            break :blk n;
        };
        assert(0 < N);

        const values = self.tensor_allocator.allocTensor(data_type, N, self.stream);

        const sizes = UT.dupe(dimensions, allocator);
        const strides = UT.alloc(SizeType, sizes.len, allocator);

        TC.computeStrides(sizes, strides);

        return self.tensorFromComponents(class, values, sizes, strides) catch @panic("Failed to add tensor from components.");
    }

    ////////////////////////////////////////////
    // public api for making leaf tensors

    pub fn tensor(
        self: *Graph,
        comptime class: enum { inp, wgt },
        comptime data_type: ScalarTag,
        dimensions: anytype, // either fixed-length array or slice
    ) LeafTensor(ScalarTag.asType(data_type), userTensor(class)) {
        const _class = comptime userTensor(class);

        const _data_type = comptime ScalarTag.asType(data_type);

        return self.constructTensor(_class, dimensions[0..], _data_type, self.leaf_arena.allocator());
    }

    fn adjustDependencies(self: *Graph, class: TensorClass, idx: IndexType, direction: i8) i8 {
        const array = if (class == .hid) &self.nodes.dependencies else &self.leaves.dependencies;

        std.debug.assert(@abs(direction) <= 1);

        array.items[idx] -= direction;

        return array.items[idx];
    }

    fn freeGradient(self: *Graph, comptime data_type: type, class: TensorClass, idx: IndexType) void {
        const grads_array = if (class == .hid) &self.nodes.grads else &self.leaves.grads;
        if (grads_array.items[idx]) |grads| {
            self.tensor_allocator.freeTensor(getSlice(data_type, grads), self.stream);
            grads_array.items[idx] = null;
        }
    }

    fn freeGradientRaw(self: *Graph, raw: SliceUnion, class: TensorClass, idx: IndexType) void {
        switch (raw) {
            .q8 => self.freeGradient(SC.q8, class, idx),
            .r16 => self.freeGradient(SC.r16, class, idx),
            .r32 => self.freeGradient(SC.r32, class, idx),
            .r64 => self.freeGradient(SC.r64, class, idx),
            .c16 => self.freeGradient(SC.c16, class, idx),
            .c32 => self.freeGradient(SC.c32, class, idx),
            .c64 => self.freeGradient(SC.c64, class, idx),
        }
    }

    fn freeTensor(self: *Graph, comptime data_type: type, class: TensorClass, idx: IndexType) void {
        const values = if (class == .hid) &self.nodes.values.items[idx] else &self.leaves.values.items[idx];

        if (values.len() == 0) {
            return {};
        }

        const stream = if (class == .hid) self.nodes.streams.items[idx] else self.leaves.streams.items[idx];

        const field = comptime SC.scalarName(data_type);

        self.tensor_allocator.freeTensor(@field(values.*, field), stream);

        @field(values.*, field).len = 0;

        self.freeGradient(data_type, class, idx);
    }

    fn freeTensorRaw(self: *Graph, raw: SliceUnion, class: TensorClass, idx: IndexType) void {
        switch (raw) {
            .q8 => self.freeTensor(SC.q8, class, idx),
            .r16 => self.freeTensor(SC.r16, class, idx),
            .r32 => self.freeTensor(SC.r32, class, idx),
            .r64 => self.freeTensor(SC.r64, class, idx),
            .c16 => self.freeTensor(SC.c16, class, idx),
            .c32 => self.freeTensor(SC.c32, class, idx),
            .c64 => self.freeTensor(SC.c64, class, idx),
        }
    }

    // trampoline loop for reversing graph
    fn reverse(self: *Graph, idx: SizeType, cleanup: ReverseCleanup) void {
        std.debug.assert(self.mode != .eval);

        const total = self.nodes.callbacks.items.len;

        if (total == 0) return {};

        var call_stack = std.ArrayList(usize).initCapacity(self.node_arena.allocator(), idx + 1) catch @panic("Graph.reverse: Out of Memory.");
        var free_stack: std.ArrayList(usize) = undefined;

        if (cleanup == .free) {
            free_stack = std.ArrayList(usize).initCapacity(self.node_arena.allocator(), idx + 1) catch @panic("Graph.reverse: Out of Memory.");
        }

        defer {
            call_stack.deinit();
            if (cleanup == .free) {
                free_stack.deinit();
            }
        }

        // put parameter node on the top of the stack
        call_stack.appendAssumeCapacity(idx);

        const callbacks = self.nodes.callbacks.items[0..];

        while (call_stack.getLastOrNull()) |last| {
            // result tells us if we can do a depth-wise
            // traversal and wether we've hit the last
            // reverse-child node with 'stop_call'
            const result = callbacks[last].call();

            // this check causes us to do a breadth-wise
            // traversal across the nodes reverse edges
            if (result.stop_call) {
                // if there were no more edges to explore,
                // we backup by one and continue calling
                const popped = call_stack.pop();

                if (cleanup == .free) {
                    free_stack.append(popped) catch @panic("Graph.reverse: Out of Memory");
                }

                continue;
            }

            // if there is a another node we're connected to,
            // put it on the stack and continue depth-wise
            if (result.next_node) |next| {
                if (!self.attached(next)) {
                    continue;
                }

                call_stack.append(next) catch @panic("Graph.reverse: Out of Memory");
            }
        }

        if (cleanup == .free) {
            // synchronize all unique streams
            var unique = std.StaticBitSet(DU.MAX_STREAMS).initEmpty();

            for (free_stack.items) |node| {
                unique.setValue(self.nodes.streams.items[node].ID, true);
            }

            for (0..DU.MAX_STREAMS) |i| {
                if (unique.isSet(i)) DU.synchronizeStream(&DU.stream_array[i].?);
            }

            for (free_stack.items) |node| {
                self.freeTensorRaw(self.nodes.values.items[node], .hid, node);
            }
        }
    }
};

////////////////////////////////
// non-interface graph functions
////////////////////////////////

pub fn enableGradient(self: *Graph, comptime T: type, class: TensorClass, idx: IndexType) []T {
    const values_array = if (class == .hid) &self.nodes.values else &self.leaves.values;
    const grads_array = if (class == .hid) &self.nodes.grads else &self.leaves.grads;

    const grads = &grads_array.items[idx];
    if (grads.* == null) {
        const size = values_array.items[idx].len();
        const grd = self.tensor_allocator.allocTensor(T, size, self.stream);
        grads.* = SliceUnion.init(grd);
        return grd;
    } else {
        return getSlice(T, grads.*.?);
    }
}

pub fn appendNode(
    self: *Graph,
    comptime FuncObj: type,
    args: anytype,
    out: anytype,
) @TypeOf(out) {
    const closure = Closure.init(FuncObj, args ++ .{out}) catch @panic("Out of Memory");

    UT.append(&self.nodes.callbacks, closure);

    comptime var i: SizeType = 0;
    inline while (comptime UT.tupleSize(@TypeOf(args)) > i) : (i += 1) {
        if (comptime isGraphTensor(@TypeOf(args[i]))) {
            _ = args[i].adjustDependencies(1);
        }
    }

    return out;
}

pub fn nodeTensor(
    self: *Graph,
    dimensions: Sizes, // fixed-length array
    comptime data_type: type,
) NodeTensor(data_type) {
    return self.constructTensor(TensorClass.hid, dimensions, data_type, self.node_arena.allocator());
}

/////////////////////////////////////////////
/////////////////////////////////////////////

fn reverseEdge(comptime func: anytype, comptime edge: SizeType, edge_tuple: anytype) ?SizeType {
    const arg = edge_tuple[edge];
    const DataType = @TypeOf(arg).DataType;
    const Class = @TypeOf(arg).Class;
    const graph = arg.ptr;

    // enable gradients if we don't have them
    if (arg.grads() == null) {
        const grd = enableGradient(graph, DataType, Class, arg.idx);
        fillSlice(DataType, grd, SC.asScalar(DataType, 0), arg.stream());
    }

    @call(.auto, func, .{arg.stream()} ++ edge_tuple);

    // this portion deals with whether or not we can
    // return the next reverse node up the tree. We
    // can only return the next node if we've gathered
    // all of it's dependencies.

    // ensure all partial-gradients are calculated
    if (arg.adjustDependencies(-1) == 0) {
        if (comptime Class == .wgt) {

            // by optimizing as we go along, we can free
            // gradients as soon as we're done using them
            graph.optimizer.update(graph.id(), arg.idx, arg.raw_values(), arg.raw_grads().?, arg.stream());

            // since we've collected all of our dependencies,
            // we check to see if we're a hidden tensor (node)
            // and return our node index if so
        } else if (comptime Class == .hid) {
            return arg.idx; // signal to keep reversing
        }
    }
    return null;
}

const ReverseNextResult = struct {
    next_node: ?SizeType,
    last_edge: ?SizeType,
};

fn reverseNext(comptime CallbackType: type, edge_tuple: anytype, last_edge: SizeType) ReverseNextResult {
    if (comptime !isStruct(CallbackType) or !isTuple(@TypeOf(edge_tuple))) {
        @compileError("reverseEdges: Function must be struct and arguments must be tuple.");
    }

    // CallbackType contains comptime fields... need an instance...
    const decls = CallbackType{};

    inline for (@typeInfo(CallbackType).Struct.fields) |field| {
        if (comptime reversePrefixed(field.name)) {
            const edge_index = @field(decls, field.name).edge_index;

            // we need to call the next reversal up, starting from,
            // zero until we've gone through every breadth-wise edge
            if (last_edge <= edge_index) {
                const next_node = reverseEdge(
                    @field(decls, field.name).callback,
                    edge_index,
                    edge_tuple,
                );

                return .{
                    .next_node = next_node,
                    .last_edge = edge_index,
                };
            }
        }
    }

    // some functions can allocate state for use
    // in the reversal process - cleanup resources
    if (comptime @hasField(CallbackType, "cleanup")) {
        @call(.auto, decls.cleanup, edge_tuple);
    }

    // this node's reversals are now exhausted
    return .{
        .next_node = null,
        .last_edge = null,
    };
}

// this function ensures we only call reversal members
fn reversePrefixed(string: []const u8) bool {
    const prefix: []const u8 = "reverse";

    if (string.len < prefix.len) {
        return false;
    }

    for (0..prefix.len) |i| {
        if (string[i] != prefix[i]) {
            return false;
        }
    }

    return true;
}

const ClosureBuffer = struct {
    // we want to hold at least 6 graph tensors
    // and a graph tensor is (ptr, idx)
    const BufferSize = @sizeOf(NodeTensor(SC.q8)) * 6;

    items: extern union {
        inplace: [BufferSize]u8 align(@alignOf(usize)),
        any_ptr: *align(@alignOf(usize)) anyopaque,
    },

    state: enum { buf, ptr },

    pub fn init(x: anytype) ClosureBuffer {
        const T = @TypeOf(x);
        var self: ClosureBuffer = undefined;

        if (@sizeOf(T) <= BufferSize) {
            const tmp: *T align(@alignOf(usize)) = @ptrCast(@alignCast(&self.items.inplace));
            tmp.* = x;
            self.state = .buf;
        } else {
            const ptr = std.heap.c_allocator.create(T) catch @panic("Closure.init: Out of memory.");

            ptr.* = x;
            self.items.any_ptr = ptr;
            self.state = .ptr;
        }
        return self;
    }

    pub fn cast(self: *ClosureBuffer, comptime T: type) *T {
        return switch (self.state) {
            .buf => @ptrCast(@alignCast(&self.items.inplace)),
            .ptr => @ptrCast(@alignCast(self.items.any_ptr)),
        };
    }

    pub fn deinit(self: *ClosureBuffer, comptime T: type) void {
        if (self.state == .ptr) {
            std.heap.c_allocator.destroy(@as(*T, @ptrCast(@alignCast(self.items.any_ptr))));
        }
    }
};

const Closure = struct {
    const Result = struct {
        next_node: ?SizeType,
        stop_call: bool,
    };

    func: *const fn (*ClosureBuffer, SizeType) ReverseNextResult,
    next_edge: SizeType = 0,
    args: ClosureBuffer,

    pub fn call(self: *Closure) Result {
        const result = self.func(&self.args, self.next_edge);

        if (result.last_edge) |edge| {
            self.next_edge += edge + 1;
        }

        return .{
            .next_node = result.next_node,
            .stop_call = result.last_edge == null,
        };
    }

    pub fn init(comptime CallbackType: type, edge_tuple: anytype) !Closure {
        const callback = struct {
            const EdgeTuple = @TypeOf(edge_tuple);

            pub fn call(args: *ClosureBuffer, next: SizeType) ReverseNextResult {
                const _edge_tuple = args.cast(EdgeTuple);

                const result = reverseNext(CallbackType, _edge_tuple.*, next);

                // usually a no-op, but potential calls free
                if (result.last_edge == null) {
                    args.deinit(EdgeTuple);
                }

                return result;
            }
        }.call;

        return .{
            .args = ClosureBuffer.init(edge_tuple),
            .func = callback,
        };
    }
};
