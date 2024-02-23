const std = @import("std");
const mem = std.mem;
const ArrayList = std.ArrayList;
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

const UT = @import("utility.zig");
const SC = @import("scalar.zig");
const TC = @import("tensor_components.zig");
const DU = @import("device_utils.zig");

const Optimizer = @import("optimizer.zig").Optimizer;
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
    switch (@typeInfo(T)) {
        .Struct => {
            if (!@hasDecl(T, "DataType")) { return false; }
            if (!@hasDecl(T, "Class")) { return false; }
            return (T == GraphTensor(T.DataType, T.Class));
        },
        else => return false,
    }
}

fn fillSlice(
    comptime T: type,
    x_slice: []T,
    value: T,
    stream: Stream,
) void {
    kernel_fill.call(.{
        stream, x_slice.ptr, value, x_slice.len
    });
    DU.synchronizeStream(stream);    
}

pub fn fill(X: anytype, value: anytype) Contract(
    isGraphTensor(@TypeOf(X)), Returns(void)
){
    const T = Child(@TypeOf(X));

    fillSlice(T, X.values(), SC.asScalar(T, value), X.ptr.stream);
}

pub const TensorClass = enum {
  inp, wgt, hid, 
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
            
        pub fn name(self: Self) []const u8 {
            return self.ptr.leaves.names.items[self.idx];
        }
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
        pub fn free(self: Self) void {
            self.ptr.freeTensor(self);
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
            .Struct => T.DataType, else => T
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
            if (self.raw_grads()) |grd| {
                return getSlice(DataType, grd);
            }
            return null;
        }
        pub fn len(self: Self) SizeType {
            return self.values().len;
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
        pub fn reverse(self: Self) void {

            if (self.grads() == null) {
                // if a loss hasn't been applied, this will be null
                const grd = self.ptr.enableGradient(DataType, .hid, self.idx);
                // derivative with respect to self is 1
                fillSlice(DataType, grd, SC.asScalar(DataType, 1), self.ptr.stream);
            }

            // call graph reverse
            self.ptr.reverse(self.idx);
        }
        fn adjustDependencies(self: Self, comptime direction: i8) i8 {
            return self.ptr.adjustDependencies(.hid, self.idx, direction);
        }
    };
}

fn GraphTensor(comptime data_type: type, comptime class: TensorClass) type {
    if (comptime class == .hid) {
        return NodeTensor(data_type);
    } else {
        return LeafTensor(data_type, class);
    }
}

const TypeOption = enum {
    q8, r16, r32, r64, c16, c32, c64,

    pub fn asType(comptime opt: TypeOption) type {
        return switch (opt) {
             .q8 => SC.q8,
            .r16 => SC.r16,
            .r32 => SC.r32,
            .r64 => SC.r64,
            .c16 => SC.c16,
            .c32 => SC.c32,
            .c64 => SC.c64,
        };
    }
    pub fn asOption(comptime T: type) TypeOption {
        return switch (T) {
             SC.q8 => TypeOption.q8,
            SC.r16 => TypeOption.r16,
            SC.r32 => TypeOption.r32,
            SC.r64 => TypeOption.r64,
            SC.c16 => TypeOption.c16,
            SC.c32 => TypeOption.c32,
            SC.c64 => TypeOption.c64,
            else => @compileError("Invalid type for asOptoin: " ++ @typeName(T)),
        };
    }
};

pub const GraphConfig = struct {
    optimizer: Optimizer,
    auto_free_wgt_grads: bool = false,
    auto_free_inp_grads: bool = false,
    auto_free_hid_nodes: bool = true,
    stream: Stream,
};

pub const Graph = struct {

    const Self = @This();

    const Leaves = struct {
        names: ArrayList([]const u8),
        classes: ArrayList(TensorClass),
        values: ArrayList(SliceUnion),
        grads: ArrayList(?SliceUnion),
        sizes: ArrayList(Sizes),
        strides: ArrayList(Strides),
        dependencies: ArrayList(i8),
        pub fn init(allocator: std.mem.Allocator, block_size: usize) !Leaves {
            return @This() {
                .names = try ArrayList([]const u8).initCapacity(allocator, block_size),
                .classes = try ArrayList(TensorClass).initCapacity(allocator, block_size),
                .values = try ArrayList(SliceUnion).initCapacity(allocator, block_size),
                .grads = try ArrayList(?SliceUnion).initCapacity(allocator, block_size),
                .sizes = try ArrayList(Sizes).initCapacity(allocator, block_size),
                .strides = try ArrayList(Strides).initCapacity(allocator, block_size),
                .dependencies = try ArrayList(i8).initCapacity(allocator, block_size),
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
        pub fn init(allocator: std.mem.Allocator, block_size: usize) !Nodes {
            return @This() {
                .callbacks = try ArrayList(Closure).initCapacity(allocator, block_size),
                .values = try ArrayList(SliceUnion).initCapacity(allocator, block_size),
                .grads = try ArrayList(?SliceUnion).initCapacity(allocator, block_size),
                .sizes = try ArrayList(Sizes).initCapacity(allocator, block_size),
                .strides = try ArrayList(Strides).initCapacity(allocator, block_size),
                .dependencies = try ArrayList(i8).initCapacity(allocator, block_size),
                .attached = try ArrayList(bool).initCapacity(allocator, block_size),
            };
        }
    };

    node_count: usize,
    leaf_count: usize,

    leaf_arena: std.heap.ArenaAllocator,
    node_arena: std.heap.ArenaAllocator,

    tensor_allocator: LaneAllocator,

    // optimizer interface to update weights
    optimizer: Optimizer,

    // tensor stuff...
    leaves: Leaves,
    nodes: Nodes,

    // free upon reversal
    auto_free_wgt_grads: bool,
    auto_free_inp_grads: bool,
    auto_free_hid_nodes: bool,

    stream: Stream,

    // this function ensures the correct allocator
    // and size are passed to the designated block
    fn initLeaves(self: *Self) Leaves {
        return Leaves.init(self.leaf_arena.allocator(), self.leaf_count)
            catch @panic("Graph.initLeaves: Out of Memory");
    }

    // this function ensures the correct allocator
    // and size are passed to the designated block
    fn initNodes(self: *Self) Nodes {
        return Nodes.init(self.node_arena.allocator(), self.node_count)
            catch @panic("Graph.initNodes: Out of Memory");
    }
 
    pub fn init(config: GraphConfig) *Self {

        const self: *Self = std.heap.c_allocator.create(Self)
            catch @panic("Graph.init: Out of Memory");
            
        self.optimizer = config.optimizer;

        self.stream = config.stream;

        self.leaf_count = 0;
        self.node_count = 0;

        self.leaf_arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
        self.node_arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
            
        self.leaves = self.initLeaves();
        self.nodes = self.initNodes();

        self.tensor_allocator.setup();

        self.auto_free_wgt_grads = config.auto_free_wgt_grads;
        self.auto_free_inp_grads = config.auto_free_inp_grads;
        self.auto_free_hid_nodes = config.auto_free_hid_nodes;

        return self;
    }

    pub fn deinit(self: *Self) void {
        self.node_arena.deinit();
        self.leaf_arena.deinit();

        self.tensor_allocator.deinit(self.stream);

        // this must always be last
        std.heap.c_allocator.destroy(self);
    }

    const ResetMode = enum {
        all, node, leaf
    };

    // this function helps defragment the arenas and
    // preps the graph - called by reverse conditionally
    pub fn reset(self: *Self, mode: ResetMode) void {
        if (mode == .all or mode == .leaf) {
            // drop all of our values out of play
            _ = self.leaf_arena.reset(.retain_capacity);

            // reinstantiate the array lists
            self.leaves = self.initLeaves();

            // this will stack up otherwise
            self.leaf_count = 0;
        }
        if (mode == .all or mode == .node) {
            // drop all of our values out of play
            _ = self.node_arena.reset(.retain_capacity);

            // reinstantiate the array lists
            self.nodes = self.initNodes();

            // this will stack up otherwise
            self.node_count = 0;
        }
    }

    fn tensorFromComponents(
        self: *Self, 
        name: ?[]const u8,
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
            self.node_count += 1;
        } else {
            if (name) |_name| {
                assert(_name.len != 0);
                try self.leaves.names.append(_name);
                try self.leaves.classes.append(class);
                try self.leaves.values.append(SliceUnion.init(values));
                try self.leaves.sizes.append(sizes);
                try self.leaves.strides.append(strides);
                try self.leaves.grads.append(null);
                try self.leaves.dependencies.append(0);            
                self.leaf_count += 1;
            } else {
                @panic("Leaf tensors cannot have null names.");
            }
        }

        const idx = if (class == .hid)
            self.nodes.values.items.len - 1 else self.leaves.values.items.len - 1;

        return GraphTensor(Child(@TypeOf(values)), class) { 
            .ptr = self, .idx = idx,
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
        self: *Self, 
        name: ?[]const u8,
        comptime class: TensorClass, 
        dimensions: Sizes, // fixed-length array
        comptime data_type: type,
        allocator: std.mem.Allocator
    ) GraphTensor(data_type, class) {

        assert(0 < dimensions.len);

        const N = blk: {
            var n: SizeType = 1;
            for (dimensions) |s| { n *= s; }
            break :blk n;
        };
        assert(0 < N);

        const values = self.tensor_allocator.allocTensor(data_type, N, self.stream);

        const sizes = UT.dupe(dimensions, allocator);
        const strides = UT.alloc(SizeType, sizes.len, allocator);

        TC.computeStrides(sizes, strides);

        return self.tensorFromComponents(name, class, values, sizes, strides)
            catch @panic("Failed to add tensor from components.");
    }

    ////////////////////////////////////////////
    // internal api for making node tensors

    pub fn nodeTensor(
        self: *Self, 
        dimensions: Sizes, // fixed-length array
        comptime data_type: type,
    ) NodeTensor(data_type) {
        return self.constructTensor(
            null, TensorClass.hid, dimensions, data_type, self.node_arena.allocator()
        );
    }

    ////////////////////////////////////////////
    // public api for making leaf tensors

    pub fn tensor(
        self: *Self, 
        name: []const u8,
        comptime class: enum { inp, wgt }, 
        comptime data_type: TypeOption,
        dimensions: anytype, // either fixed-length array or slice
    ) LeafTensor(TypeOption.asType(data_type), userTensor(class)) {

        const _class = comptime userTensor(class);

        const _data_type = comptime TypeOption.asType(data_type);

        return self.constructTensor(
            name, _class, dimensions[0..], _data_type, self.leaf_arena.allocator()
        );
    }


    fn enableGradient(self: *Self, comptime T: type, class: TensorClass, idx: IndexType) []T {

        const values_array = if (class == .hid) 
            &self.nodes.values else &self.leaves.values;

        const grads_array = if (class == .hid) 
            &self.nodes.grads else &self.leaves.grads;
        
        const grads: *?SliceUnion = &grads_array.items[idx];
        if (grads.* == null) {            
            const size = values_array.items[idx].len();
            const grd = self.tensor_allocator.allocTensor(T, size, self.stream);
            grads.* = SliceUnion.init(grd);
            return grd;
        } else {
            return getSlice(T, grads.*.?);
        }
    }

    fn disableGradient(self: *Self, comptime data_type: type, class: TensorClass, idx: IndexType) void {
        const grads_array = if (class == .hid) 
            &self.nodes.grads else &self.leaves.grads;

        if (grads_array.items[idx]) |grads| {            
            self.tensor_allocator.freeTensor(getSlice(data_type, grads), self.stream);
            grads_array.items[idx] = null;
        }
    }

    fn freeTensor(self: *Self, X: anytype) void {
        const T = @TypeOf(X);
        self.tensor_allocator.freeTensor(X.values(), self.stream);
        self.disableGradient(T.DataType, T.Class, X.idx);
    }

    fn adjustDependencies(self: *Self, class: TensorClass, idx: IndexType, direction: i8) i8 {
        const array = if (class == .hid) 
            &self.nodes.dependencies else &self.leaves.dependencies;

        std.debug.assert(@abs(direction) <= 1);

        array.items[idx] -= direction;
        
        return array.items[idx];
    }

    // TODO: 
    //    I don't like that this function is public
    //    but it's used by the tensor ops to make
    //    nodes on the graph for a given op.
    pub fn appendNode(
        self: *Self, 
        comptime FuncObj: type,
        args: anytype,
        out: anytype,
    ) @TypeOf(out) {
        const closure = Closure.init(FuncObj, args ++ .{ out }) 
            catch @panic("Out of Memory");
        
        UT.append(&self.nodes.callbacks, closure);

        comptime var i: SizeType = 0;
        inline while (comptime UT.tupleSize(@TypeOf(args)) > i) : (i += 1) {
            if (comptime isGraphTensor(@TypeOf(args[i]))) {
                _ = args[i].adjustDependencies(1);
            }
        }
        return out;
    }

    // trampoline loop for reversing graph
    fn reverse(self: *Self, idx: SizeType) void {

        const total = self.nodes.callbacks.items.len;

        if (total == 0) return;

        var stack = std.ArrayListUnmanaged(usize).initCapacity(self.node_arena.allocator(), idx + 1)
            catch @panic("Graph.reverse: Out of Memory.");

        defer stack.deinit(self.node_arena.allocator());

        // put parameter node on the top of the stack
        stack.appendAssumeCapacity(idx);

        const callbacks = self.nodes.callbacks.items[0..];

        while (stack.getLastOrNull()) |last| {

            // result tells us if we can do a depth-wise
            // traversal and wether we've hit the last
            // reverse-child node with 'stop_call'
            const result = callbacks[last].call();

            // this check causes us to do a breadth-wise
            // traversal across the nodes reverse edges
            if (result.stop_call) { 

                // if there were no more edges to explore,
                // we backup by one and continue calling
                _ = stack.pop();

                continue;
            }
                    
            // if there is a another node we're connected to,
            // put it on the stack and continue depth-wise
            if (result.next_node) |next| {

                if (!self.attached(next))
                    continue;

                stack.appendAssumeCapacity(next);
            }
        }

        // if we reversed the last index, all callbacks are exhausted
        if (self.auto_free_hid_nodes and (total == idx + 1)) {
            self.reset(.node);
        }
    }
};


 /////////////////////////////////////////////
/////////////////////////////////////////////

fn reverseEdge(
    comptime func: anytype,
    comptime edge: SizeType,
    edge_tuple: anytype,
) ?SizeType {

    const arg = edge_tuple[edge];
    const DataType = @TypeOf(arg).DataType;
    const Class = @TypeOf(arg).Class;
    const graph_ptr = arg.ptr;

    // why calculate the input gradient if we are
    // freeing it immediately after creating it?
    if ((Class == .inp) and graph_ptr.auto_free_inp_grads) {
        return null;
    }

    // enable gradients if we don't have them
    if (arg.grads() == null) {
        const grd = graph_ptr.enableGradient(DataType, Class, arg.idx);
        fillSlice(DataType, grd, SC.asScalar(DataType, 0), graph_ptr.stream);
    }

    @call(.auto, func, edge_tuple);   

    // this portion deals with whether or not we can
    // return the next reverse node up the tree. We
    // can only return the next node if we've gathered
    // all of it's dependencies.

    // ensure all partial-gradients are calculated
    if (arg.adjustDependencies(-1) == 0) {

        if (comptime Class == .wgt) {

            // by optimizing as we go along, we can free
            // gradients as soon as we're done using them
            graph_ptr.optimizer.update(
                arg.name(), arg.raw_values(), arg.raw_grads().?
            );

            if (graph_ptr.auto_free_wgt_grads) {
                graph_ptr.disableGradient(DataType, Class, arg.idx);
            }

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

fn reverseNext(
        comptime CallbackType: type,
        edge_tuple: anytype,
        last_edge: SizeType
    ) ReverseNextResult {

    if (comptime !isStruct(CallbackType) or !isTuple(@TypeOf(edge_tuple))){
        @compileError("reverseEdges: Function must be struct and arguments must be tuple.");
    }
    
    // CallbackType contains comptime fields... need an instance...
    const decls = CallbackType{ };

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

                return ReverseNextResult {
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
    
    // the last argument is always the reverse-parent node.
    const N = comptime UT.tupleSize(@TypeOf(edge_tuple)) - 1;

    const graph_ptr = edge_tuple[N].ptr;

    // Free unnecessary data as we go along...
    if (graph_ptr.auto_free_hid_nodes) {
        // we need to synchronize here to prevent the parent
        // node from being released before the other children
        // have a chance to collect their gradients
        DU.synchronizeStream(edge_tuple[0]);

        graph_ptr.freeTensor(edge_tuple[N]);
    }

    // this node's reversals are now exhausted
    return ReverseNextResult {
        .next_node = null,
        .last_edge = null,
    };
}

// this function ensures we only call reversal members
fn reversePrefixed(string: []const u8) bool {
    const prefix: []const u8 = "reverse";

    if (string.len < prefix.len) 
        return false;

    for (0..prefix.len) |i| {
        if (string[i] != prefix[i]) return false;
    }
    return true;
}


const ClosureBuffer = struct {

    const Self = @This();

    // we want to hold at least 4 graph tensors and a stream
    // and a graph tensor is (ptr, idx)
    const BufferSize = @sizeOf(usize) * 8 + @sizeOf(usize);

    items: extern union {
        inplace: [BufferSize]u8 align(@alignOf(usize)),  
        any_ptr: *align(@alignOf(usize)) anyopaque,
    },
    
    state: enum { buf, ptr },

    pub fn init(x: anytype) Self {
        const T = @TypeOf(x);
        var self: ClosureBuffer = undefined;

        if (@sizeOf(T) <= BufferSize) {

            const tmp: *T align(@alignOf(usize)) = @ptrCast(@alignCast(&self.items.inplace));
            tmp.* = x;
            self.state = .buf;

        } else {

            const ptr = std.heap.c_allocator.create(T)
                catch @panic("Closure.init: Out of memory.");

            ptr.* = x;
            self.items.any_ptr = ptr;
            self.state = .ptr;
        }
        return self;
    }

    pub fn cast(self: *Self,  comptime T: type) *T {
        return switch(self.state) {
            .buf => @ptrCast(@alignCast(&self.items.inplace)),
            .ptr => @ptrCast(@alignCast(self.items.any_ptr)),
        };
    }

    pub fn deinit(self: *Self, comptime T: type) void {
        if (self.state == .ptr) {
            std.heap.c_allocator.destroy(
                @as(*T, @ptrCast(@alignCast(self.items.any_ptr)))
            );
        }
    }
};


const ClosureResult = struct {
    next_node: ?SizeType,
    stop_call: bool
};

const Closure = struct {

    const Self = @This();

    args: ClosureBuffer,
    func: *const fn(*ClosureBuffer, SizeType) ReverseNextResult,
    next_edge: SizeType = 0,

    pub fn call(self: *Self) ClosureResult {
        const result = self.func(&self.args, self.next_edge);      

        if (result.last_edge) |edge| {
            self.next_edge += edge + 1;
        }

        return ClosureResult {
            .next_node = result.next_node,  
            .stop_call = (result.last_edge == null)
        };
    }

    pub fn init(
        comptime CallbackType: type,
        edge_tuple: anytype,
    ) !Self {    

        const callback = struct {
            
            const EdgeTuple = @TypeOf(edge_tuple);

            pub fn call(args: *ClosureBuffer, next: SizeType) ReverseNextResult {

                const _edge_tuple = args.cast(EdgeTuple);

                const result = reverseNext(
                    CallbackType, _edge_tuple.*, next
                );

                // usually a no-op, but potentiall calls free
                if (result.last_edge == null) {
                    args.deinit(EdgeTuple);
                }

                return result;
            }
        }.call;

        return Closure { 
            .args = ClosureBuffer.init(edge_tuple), 
            .func = callback, 
        };
    }
};

