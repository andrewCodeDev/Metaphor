const mp = @import("metaphor");
const EU = @import("example_utils.zig");

pub fn main() !void {
    // To begin, we'll setup 3 tensors { A, x, b }
    // for this example, we'll use a Rank-2 tensor
    // for A and Rank-1 tensors for x and b.

    mp.device.init(0);

    const stream = mp.stream.init();
    defer mp.stream.deinit(stream);

    const G = mp.Graph.init(.{ .stream = stream, .mode = .eval });
    defer G.deinit();

    // square matrix for the sake of demonstration
    const M: usize = 32;
    const N: usize = 32;

    /////////////////////////////////////////////////////

    const x = G.tensor(.inp, .r32, mp.Rank(1){M});
    const A = G.tensor(.inp, .r32, mp.Rank(2){ M, N });
    const b = G.tensor(.inp, .r32, mp.Rank(1){N});

    mp.mem.randomize(x, .gauss);
    mp.mem.randomize(A, .gauss);
    mp.mem.randomize(b, .gauss);

    // in this example, we'll explore using the innerProduct
    // and linear functions. Linear transformations have the
    // following form:
    //
    //    alpha: scalar
    //    beta: scalar
    //    A: tensor
    //    x: tensor
    //    b: tensor
    //    y: tensor
    //
    //    y = alpha * A.x + beta * b
    //
    // Linear transformations have the following requirements:
    //
    //    1) A's dimensions are compatible with x's for inner products
    //    2) The product of A.x results in the same dimensions for b
    //    3) b's dimensions are equal to y's dimensions
    //
    // If a single inner product without the addition to b is needed,
    // set beta=0.0 and pass in y as the parameter for b. This guarantees
    // requirement 3 and the linear kernel will skip the addition where
    // the coefficient beta is equal to 0.0 within a small epsilon.

    /////////////////////////////////////////////////////
    {

        // inner product, beta=0.0
        const x_A = mp.ops.innerProduct(x, A, "i,ij->j");
        const A_x = mp.ops.innerProduct(A, x, "ij,j->i");

        // linear, beta=1.0
        const x_A_b = mp.ops.linear(x, A, b, "i,ij->j");
        const A_x_b = mp.ops.linear(A, x, b, "ij,j->i");

        // to check our work, we can allocate some memory to the
        // cpu and perform the naive versions of these functions
        const x_cpu = try EU.copyToCPU(x.values(), stream);
        defer EU.freeCPU(x_cpu);

        const A_cpu = try EU.copyToCPU(A.values(), stream);
        defer EU.freeCPU(A_cpu);

        const b_cpu = try EU.copyToCPU(b.values(), stream);
        defer EU.freeCPU(b_cpu);

        const y_cpu = try EU.allocCPU(mp.scalar.r32, N);
        defer EU.freeCPU(y_cpu);

        // check x.A, x.A + b
        {
            EU.cpuMatmul(x_cpu, A_cpu, y_cpu, 1, M, N);

            const x_A_cpu = try EU.copyToCPU(x_A.values(), stream);
            defer EU.freeCPU(x_A_cpu);

            EU.verifyResults("x_A", x_A_cpu, y_cpu);

            // now check with added bias
            const x_A_b_cpu = try EU.copyToCPU(x_A_b.values(), stream);
            defer EU.freeCPU(x_A_b_cpu);

            EU.cpuAdd(x_A_cpu, b_cpu, y_cpu);

            EU.verifyResults("x_A_b", x_A_b_cpu, y_cpu);
        }

        // check A.x
        {
            EU.cpuMatmul(A_cpu, x_cpu, y_cpu, M, N, 1);

            const A_x_cpu = try EU.copyToCPU(A_x.values(), stream);
            defer EU.freeCPU(A_x_cpu);

            EU.verifyResults("A_x", A_x_cpu, y_cpu);

            // now check with added bias
            const A_x_b_cpu = try EU.copyToCPU(A_x_b.values(), stream);
            defer EU.freeCPU(A_x_b_cpu);

            EU.cpuAdd(A_x_cpu, b_cpu, y_cpu);

            EU.verifyResults("A_x_b", A_x_b_cpu, y_cpu);
        }
    }

    ////////////////////////////////////////////
    // Now let's do matrices -------------------

    G.reset(.node, .all);

    const B = G.tensor(.inp, .r32, mp.Rank(2){ N, M });

    mp.mem.randomize(B, .gauss);
 
    {
        // A.B
        const A_B = mp.ops.innerProduct(A, B, "ij,jk->ik");

        // T(A).B
        const AT_B = mp.ops.innerProduct(A, B, "ji,jk->ik");

        // A.T(B)
        const A_BT = mp.ops.innerProduct(A, B, "ij,kj->ik");

        mp.stream.synchronize(stream);

        const A_cpu = try EU.copyToCPU(A.values(), stream);
        const B_cpu = try EU.copyToCPU(B.values(), stream);
        const T_cpu = try EU.allocCPU(mp.scalar.r32, N * M);
        const C_cpu = try EU.allocCPU(mp.scalar.r32, N * M);

        {
            const A_B_cpu = try EU.copyToCPU(A_B.values(), stream);
            defer EU.freeCPU(A_B_cpu);
            EU.cpuMatmul(A_cpu, B_cpu, C_cpu, M, N, N);
            EU.verifyResults("A.B", A_B_cpu, C_cpu);
        }
        {
            const AT_B_cpu = try EU.copyToCPU(AT_B.values(), stream);
            defer EU.freeCPU(AT_B_cpu);
            EU.cpuTranspose(A_cpu, T_cpu, M, N);
            EU.cpuMatmul(T_cpu, B_cpu, C_cpu, M, N, N);
            EU.verifyResults("T(A).B", AT_B_cpu, C_cpu);
        }
        {
            const A_BT_cpu = try EU.copyToCPU(A_BT.values(), stream);
            defer EU.freeCPU(A_BT_cpu);
            EU.cpuTranspose(B_cpu, T_cpu, M, N);
            EU.cpuMatmul(A_cpu, T_cpu, C_cpu, M, N, N);
            EU.verifyResults("A.T(B)", A_BT_cpu, C_cpu);
        }
    }

    mp.device.check();
    ////////////////////////////////////////////
}
