const mp = @import("metaphor");
const EU = @import("example_utils.zig");

pub fn main() !void {

    mp.device.init(0);

    const stream = mp.stream.init();
        
    defer mp.stream.deinit(stream);

    const G = mp.Graph.init(.{
        .optimizer = mp.null_optimizer,
        .auto_free_wgt_grads = false,
        .auto_free_inp_grads = false,
        .auto_free_hid_nodes = false,
        .stream = stream,
        .mode = .eval
    });

    defer G.deinit();

    const row_x: usize = 16;
    const col_x: usize = 32;

    /////////////////////////////////////////////////////

    const X1 = G.tensor("X1", .wgt, .r32, mp.Rank(1){ row_x });  
        defer X1.free();

    const X2 = G.tensor("X2", .wgt, .r32, mp.Rank(2){ row_x, col_x });  
        defer X2.free();

    mp.mem.sequence(X1, 0.0, 1.0);
    mp.mem.sequence(X2, 0.0, 1.0);

    /////////////////////////////////////////////////////

    const Z1 = mp.ops.innerProduct(X1, X2, "i,ij->j");

    Z1.reverse();

    try EU.copyAndPrintMatrix("X2: value", X2.values(),  row_x, col_x, stream);
    try EU.copyAndPrintMatrix("X1: grads", X1.grads().?,     1, row_x, stream);

    try EU.copyAndPrintMatrix("X1: value", X1.values(),      1, row_x, stream);
    try EU.copyAndPrintMatrix("X2: grads", X2.grads().?, row_x, col_x, stream);

    ////////////////////////////////////////////
}
