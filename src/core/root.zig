
// core.root:
//
// This file is exported to the other Zig-based source
// groups and to the primary metaphor.zig file
//

pub const C = @import("cimport.zig");

// imports non-public graph API
pub const cg = @import("graph.zig");
pub const Graph = cg.Graph;
pub const Tensor = cg.Tensor;

////////////////////////////////
// non-interface graph functions
////////////////////////////////

pub const enable_gradient = cg.enable_gradient;
pub const adjust_dependencies = cg.adjust_dependencies;
pub const attach_op = cg.attach_op;
pub const derive = cg.derive;
pub const dkey = cg.dkey;

// op-interface components
pub const OpArgs = cg.OpArgs;
pub const OpDatum = cg.OpDatum;
pub const ArgIterator = cg.ArgIterator;
pub const OpInterface = cg.OpInterface;

pub const scalar = @import("scalar.zig");
pub const utils = @import("utils.zig");
pub const kernels = @import("kernels.zig");

pub const invoke = @import("common.zig").invoke;
