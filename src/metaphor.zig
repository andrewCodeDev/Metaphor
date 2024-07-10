const std = @import("std");
const assert = std.debug.assert;

const core = @import("core");

pub const Graph = core.Graph;
pub const Tensor = core.Tensor;
pub const scalar = core.scalar;

///////////////////////////////////////////////
///////////////////////////////////////////////

pub const device = struct {
    pub const init = core.utils.init_device;
    pub const sync = core.utils.synchronize_device;
    pub const check = core.utils.check_last_error;
    pub const total_memory = core.utils.total_memory;
};

pub const stream = struct {
    pub const init = core.utils.init_stream;
    pub const deinit = core.utils.deinit_stream;
    pub const sync = core.utils.synchronize_stream;
    pub const Group = core.utils.StreamGroup;
};

pub const util = struct {
    pub const to_device = core.utils.copy_to_device;
    pub const from_device = core.utils.copy_from_device;
    pub const to_device_raw = core.utils.copy_to_device_raw;
    pub const from_device_raw = core.utils.copy_from_device_raw;
    pub const alloc = core.utils.alloc;
    pub const create = core.utils.create;
    pub const free = core.utils.free;
};

pub const algo = struct {
    const mem = @import("algo/mem.zig");
    pub const fill = mem.fill;
    pub const copy = mem.copy;
    pub const sequence = mem.sequence;
};

pub const ops = struct {
    pub const add = @import("ops/addition.zig").forward;
    pub const sub = @import("ops/subtraction.zig").forward;
    pub const hadamard = @import("ops/hadamard.zig").forward;
    pub const translate = @import("ops/translate.zig").forward;
    pub const dilate = @import("ops/dilate.zig").forward;
    pub const relu = @import("ops/relu.zig").forward;
    pub const reduce = @import("ops/reduce.zig").forward;
    pub const broadcast = @import("ops/broadcast.zig").forward;
    pub const inner_product = @import("ops/inner_product.zig").forward;
    pub const inner_product_scaled = @import("ops/inner_product.zig").forward_scaled;
    pub const permutate = @import("ops/permutate.zig").forward;
};
