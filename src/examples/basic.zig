const mp = @import("metaphor");
const EU = @import("example_utils.zig");

//TODO: actually explain the basics here

pub fn main() !void {

    mp.device.init(0);

    const stream = mp.stream.init();
        
    defer mp.stream.deinit(stream);

    const G = mp.Graph.init(.{
        .optimizer = mp.null_optimizer,
        .stream = stream,
        .mode = .train
    });

    defer G.deinit();

    const row_x: usize = 16;
    const col_x: usize = 32;

    /////////////////////////////////////////////////////

    const X1 = G.tensor(.wgt, .r32, mp.Rank(1){ row_x });  
    const X2 = G.tensor(.wgt, .r32, mp.Rank(2){ row_x, col_x });  

    mp.mem.sequence(X1, 0.0, 1.0);
    mp.mem.sequence(X2, 0.0, 1.0);

    /////////////////////////////////////////////////////

    const Z1 = mp.ops.innerProduct(X1, X2, "i,ij->j");

    Z1.reverse();

    //try EU.copyAndPrintMatrix("X2: value", X2.values(),  row_x, col_x, stream);
    //try EU.copyAndPrintMatrix("X1: grads", X1.grads().?,     1, row_x, stream);

    //try EU.copyAndPrintMatrix("X1: value", X1.values(),      1, row_x, stream);
    //try EU.copyAndPrintMatrix("X2: grads", X2.grads().?, row_x, col_x, stream);

    ////////////////////////////////////////////
}
