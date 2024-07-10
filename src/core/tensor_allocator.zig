const std = @import("std");
const builtin = @import("builtin");
const utils = @import("utils.zig");
const dev = utils.dev;
const Stream = utils.Stream;

const SC = @import("scalar.zig");
const SizeType = @import("tensor_components.zig").SizeType;
const TensorData = @import("tensor_components.zig").TensorData;

const MAX_TYPES: usize = 3;
const MAX_NODES = 1024;
const CACHE_SIZE = 512;

const Self = @This();
const AnyList = std.SinglyLinkedList(*anyopaque);
const AnyNode = AnyList.Node;

//// data cache...
scalar_lanes: [MAX_TYPES]AnyList,
tensor_bytes: [CACHE_SIZE]usize,
tensor_nodes: [CACHE_SIZE]AnyList,
node_index: usize,

// node cache...
free_nodes: AnyList,
node_buffer: [MAX_NODES]AnyNode,

// Has to be called for the allocator to work... not quite init because
// we can't return stack memory otherwise the nodes will point to invalid
// addresses once we've exited the init function.

pub fn setup(self: *Self) void {

    // this data structure gets created using an allocator.
    // due to that, we have to set the default field values.

    // n0 -> n1 -> n2 -> ... -> n_i -> null

    for (0..MAX_NODES - 1) |i| {
        self.node_buffer[i].next = &self.node_buffer[i + 1];
        self.node_buffer[i].data = sentinel_ptr();
    }

    self.node_buffer[MAX_NODES - 1].next = null;
    self.node_buffer[MAX_NODES - 1].data = sentinel_ptr();

    // l.first -> n0...
    self.free_nodes.first = &self.node_buffer[0];

    for (self.scalar_lanes[0..]) |*list| {
        list.first = null;
    }

    @memset(self.tensor_bytes[0..], 0);

    self.node_index = 0;
}

pub fn deinit(self: *Self, stream: Stream) void {
    // all nodes reference the elements of
    // the free node buffer. Free anything
    // that hasn't been marked as sentinel
    for (self.node_buffer[0..]) |node| {
        if (node.data != sentinel_ptr()) utils.free_raw(node.data, stream);
    }
}

/// allocates a scalar value for reductions or storing important values
pub fn create(self: *Self, dtype: SC.Tag, stream: Stream) *anyopaque {

    const lane = @intFromEnum(dtype);

    if (self.scalar_lanes[lane].popFirst()) |node|
        return self.release_data_and_cache_node(node);

    return utils.alloc_raw(dtype.size(), stream);
}

/// frees a scalar value - caches if free nodes are available
pub fn destroy(self: *Self, dtype: SC.Tag, scalar: *anyopaque, stream: Stream) void {

    var node = self.get_free_node() orelse {
        return utils.free_raw(scalar, stream);
    };

    node.data = scalar;

    const lane = @intFromEnum(dtype); 

    self.scalar_lanes[lane].prepend(node);
}

/// allocates a tensor - retrieves from cache if size can be accomodated
pub fn alloc(self: *Self, dtype: SC.Tag, N: usize, stream: Stream) TensorData {

    const byte_len = N * dtype.size();

    if (std.mem.indexOfScalar(usize, self.tensor_bytes[0..self.node_index], byte_len)) |i| {

        if (self.tensor_nodes[i].popFirst()) |node| {

            const ptr = self.release_data_and_cache_node(node);

            return TensorData.init(dtype, ptr, N);
        } 
    }

    const raw = utils.alloc_raw(byte_len, stream);
    
    return TensorData.init(dtype, raw, N);
}

/// frees a tensor - caches if free nodes are available
pub fn free(self: *Self, tensor: TensorData, stream: Stream) void {

    const raw = tensor.ptr.raw();

    // exhausted node cache - free and return
    var node = self.get_free_node() orelse {
        return utils.free_raw(raw, stream);
    };

    node.data = raw;

    const byte_len = tensor.len * tensor.dtype().size();

    if (std.mem.indexOfScalar(usize, self.tensor_bytes[0..self.node_index], byte_len)) |i| {
        return self.tensor_nodes[i].prepend(node);
    }

    if (self.node_index < CACHE_SIZE) {
        defer self.node_index += 1;
        self.tensor_bytes[self.node_index] = byte_len;
        self.tensor_nodes[self.node_index] = .{ .first = node };
        return;
    }

    // no tensor blocks were found that match the length
    utils.free_raw(raw, stream);
}

/// calculate the number of total used nodes
pub fn used(self: *const Self) usize {
    var count: usize = 0;
    for (self.node_buffer[0..]) |node| {
        if (node.data != sentinel_ptr()) count += 1;
    }
    return count;
}

///////////////////////////////////////////
// internal functions for the TensorAllocator

fn get_free_node(self: *Self) ?*AnyNode {
    const node = self.free_nodes.popFirst() orelse return null;
    node.next = null;
    return node;
}

// we have to capture the data pointer before breaking the node's connection
fn release_data_and_cache_node(self: *Self, node: *AnyNode) *anyopaque {
    const ptr = node.data;
    node.data = sentinel_ptr();
    self.free_nodes.prepend(node);
    return ptr;
}

// this is used to default assign to node data. Probably not
// necessary and currently not used for anything important
inline fn sentinel_ptr() *anyopaque {
    return @ptrFromInt(std.math.maxInt(usize));
}
