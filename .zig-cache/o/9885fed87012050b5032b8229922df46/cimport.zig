pub const __builtin_bswap16 = @import("std").zig.c_builtins.__builtin_bswap16;
pub const __builtin_bswap32 = @import("std").zig.c_builtins.__builtin_bswap32;
pub const __builtin_bswap64 = @import("std").zig.c_builtins.__builtin_bswap64;
pub const __builtin_signbit = @import("std").zig.c_builtins.__builtin_signbit;
pub const __builtin_signbitf = @import("std").zig.c_builtins.__builtin_signbitf;
pub const __builtin_popcount = @import("std").zig.c_builtins.__builtin_popcount;
pub const __builtin_ctz = @import("std").zig.c_builtins.__builtin_ctz;
pub const __builtin_clz = @import("std").zig.c_builtins.__builtin_clz;
pub const __builtin_sqrt = @import("std").zig.c_builtins.__builtin_sqrt;
pub const __builtin_sqrtf = @import("std").zig.c_builtins.__builtin_sqrtf;
pub const __builtin_sin = @import("std").zig.c_builtins.__builtin_sin;
pub const __builtin_sinf = @import("std").zig.c_builtins.__builtin_sinf;
pub const __builtin_cos = @import("std").zig.c_builtins.__builtin_cos;
pub const __builtin_cosf = @import("std").zig.c_builtins.__builtin_cosf;
pub const __builtin_exp = @import("std").zig.c_builtins.__builtin_exp;
pub const __builtin_expf = @import("std").zig.c_builtins.__builtin_expf;
pub const __builtin_exp2 = @import("std").zig.c_builtins.__builtin_exp2;
pub const __builtin_exp2f = @import("std").zig.c_builtins.__builtin_exp2f;
pub const __builtin_log = @import("std").zig.c_builtins.__builtin_log;
pub const __builtin_logf = @import("std").zig.c_builtins.__builtin_logf;
pub const __builtin_log2 = @import("std").zig.c_builtins.__builtin_log2;
pub const __builtin_log2f = @import("std").zig.c_builtins.__builtin_log2f;
pub const __builtin_log10 = @import("std").zig.c_builtins.__builtin_log10;
pub const __builtin_log10f = @import("std").zig.c_builtins.__builtin_log10f;
pub const __builtin_abs = @import("std").zig.c_builtins.__builtin_abs;
pub const __builtin_labs = @import("std").zig.c_builtins.__builtin_labs;
pub const __builtin_llabs = @import("std").zig.c_builtins.__builtin_llabs;
pub const __builtin_fabs = @import("std").zig.c_builtins.__builtin_fabs;
pub const __builtin_fabsf = @import("std").zig.c_builtins.__builtin_fabsf;
pub const __builtin_floor = @import("std").zig.c_builtins.__builtin_floor;
pub const __builtin_floorf = @import("std").zig.c_builtins.__builtin_floorf;
pub const __builtin_ceil = @import("std").zig.c_builtins.__builtin_ceil;
pub const __builtin_ceilf = @import("std").zig.c_builtins.__builtin_ceilf;
pub const __builtin_trunc = @import("std").zig.c_builtins.__builtin_trunc;
pub const __builtin_truncf = @import("std").zig.c_builtins.__builtin_truncf;
pub const __builtin_round = @import("std").zig.c_builtins.__builtin_round;
pub const __builtin_roundf = @import("std").zig.c_builtins.__builtin_roundf;
pub const __builtin_strlen = @import("std").zig.c_builtins.__builtin_strlen;
pub const __builtin_strcmp = @import("std").zig.c_builtins.__builtin_strcmp;
pub const __builtin_object_size = @import("std").zig.c_builtins.__builtin_object_size;
pub const __builtin___memset_chk = @import("std").zig.c_builtins.__builtin___memset_chk;
pub const __builtin_memset = @import("std").zig.c_builtins.__builtin_memset;
pub const __builtin___memcpy_chk = @import("std").zig.c_builtins.__builtin___memcpy_chk;
pub const __builtin_memcpy = @import("std").zig.c_builtins.__builtin_memcpy;
pub const __builtin_expect = @import("std").zig.c_builtins.__builtin_expect;
pub const __builtin_nanf = @import("std").zig.c_builtins.__builtin_nanf;
pub const __builtin_huge_valf = @import("std").zig.c_builtins.__builtin_huge_valf;
pub const __builtin_inff = @import("std").zig.c_builtins.__builtin_inff;
pub const __builtin_isnan = @import("std").zig.c_builtins.__builtin_isnan;
pub const __builtin_isinf = @import("std").zig.c_builtins.__builtin_isinf;
pub const __builtin_isinf_sign = @import("std").zig.c_builtins.__builtin_isinf_sign;
pub const __has_builtin = @import("std").zig.c_builtins.__has_builtin;
pub const __builtin_assume = @import("std").zig.c_builtins.__builtin_assume;
pub const __builtin_unreachable = @import("std").zig.c_builtins.__builtin_unreachable;
pub const __builtin_constant_p = @import("std").zig.c_builtins.__builtin_constant_p;
pub const __builtin_mul_overflow = @import("std").zig.c_builtins.__builtin_mul_overflow;
pub const len_t = c_ulong;
pub export const WARP_SIZE: len_t = 32;
pub export const UINT_BUFFER_SIZE: len_t = 32;
pub const r16 = extern struct {
    __x: c_ushort = @import("std").mem.zeroes(c_ushort),
};
pub const r32 = f32;
pub const r64 = f64;
pub const c16 = extern struct {
    r: r16 = @import("std").mem.zeroes(r16),
    i: r16 = @import("std").mem.zeroes(r16),
};
pub const c32 = extern struct {
    r: r32 = @import("std").mem.zeroes(r32),
    i: r32 = @import("std").mem.zeroes(r32),
};
pub const c64 = extern struct {
    r: r64 = @import("std").mem.zeroes(r64),
    i: r64 = @import("std").mem.zeroes(r64),
};
pub const SortPair_r16 = extern struct {
    val: r16 = @import("std").mem.zeroes(r16),
    key: c_uint = @import("std").mem.zeroes(c_uint),
};
pub const SortPair_r32 = extern struct {
    val: r32 = @import("std").mem.zeroes(r32),
    key: c_uint = @import("std").mem.zeroes(c_uint),
};
pub const SortPair_r64 = extern struct {
    val: r64 = @import("std").mem.zeroes(r64),
    key: c_uint = @import("std").mem.zeroes(c_uint),
};
pub const RTensor16 = extern struct {
    values: [*c]r16 = @import("std").mem.zeroes([*c]r16),
    sizes: [6]len_t = @import("std").mem.zeroes([6]len_t),
    strides: [6]len_t = @import("std").mem.zeroes([6]len_t),
    dims: len_t = @import("std").mem.zeroes(len_t),
    len: len_t = @import("std").mem.zeroes(len_t),
};
pub const RTensor32 = extern struct {
    values: [*c]r32 = @import("std").mem.zeroes([*c]r32),
    sizes: [6]len_t = @import("std").mem.zeroes([6]len_t),
    strides: [6]len_t = @import("std").mem.zeroes([6]len_t),
    dims: len_t = @import("std").mem.zeroes(len_t),
    len: len_t = @import("std").mem.zeroes(len_t),
};
pub const RTensor64 = extern struct {
    values: [*c]r64 = @import("std").mem.zeroes([*c]r64),
    sizes: [6]len_t = @import("std").mem.zeroes([6]len_t),
    strides: [6]len_t = @import("std").mem.zeroes([6]len_t),
    dims: len_t = @import("std").mem.zeroes(len_t),
    len: len_t = @import("std").mem.zeroes(len_t),
};
pub const CTensor16 = extern struct {
    values: [*c]c16 = @import("std").mem.zeroes([*c]c16),
    sizes: [6]len_t = @import("std").mem.zeroes([6]len_t),
    strides: [6]len_t = @import("std").mem.zeroes([6]len_t),
    dims: len_t = @import("std").mem.zeroes(len_t),
    len: len_t = @import("std").mem.zeroes(len_t),
};
pub const CTensor32 = extern struct {
    values: [*c]c32 = @import("std").mem.zeroes([*c]c32),
    sizes: [6]len_t = @import("std").mem.zeroes([6]len_t),
    strides: [6]len_t = @import("std").mem.zeroes([6]len_t),
    dims: len_t = @import("std").mem.zeroes(len_t),
    len: len_t = @import("std").mem.zeroes(len_t),
};
pub const CTensor64 = extern struct {
    values: [*c]c64 = @import("std").mem.zeroes([*c]c64),
    sizes: [6]len_t = @import("std").mem.zeroes([6]len_t),
    strides: [6]len_t = @import("std").mem.zeroes([6]len_t),
    dims: len_t = @import("std").mem.zeroes(len_t),
    len: len_t = @import("std").mem.zeroes(len_t),
};
pub const QTensor8 = extern struct {
    values: [*c]u8 = @import("std").mem.zeroes([*c]u8),
    sizes: [6]len_t = @import("std").mem.zeroes([6]len_t),
    strides: [6]len_t = @import("std").mem.zeroes([6]len_t),
    dims: len_t = @import("std").mem.zeroes(len_t),
    len: len_t = @import("std").mem.zeroes(len_t),
};
pub const Permutation = extern struct {
    order: [6]len_t = @import("std").mem.zeroes([6]len_t),
};
pub const UintBuffer = extern struct {
    items: [32]c_uint = @import("std").mem.zeroes([32]c_uint),
    used: c_uint = @import("std").mem.zeroes(c_uint),
};
pub const StreamCtx = extern struct {
    ptr: ?*anyopaque = @import("std").mem.zeroes(?*anyopaque),
};
pub extern fn initDevice(c_uint) void;
pub extern fn mpMemAlloc(N: len_t, StreamCtx) ?*anyopaque;
pub extern fn mpMemcpyHtoD(dptr: ?*anyopaque, hptr: ?*const anyopaque, N: len_t, StreamCtx) void;
pub extern fn mpMemcpyDtoH(hptr: ?*anyopaque, dptr: ?*const anyopaque, N: len_t, StreamCtx) void;
pub extern fn mpMemFree(dptr: ?*anyopaque, StreamCtx) void;
pub extern fn mpDeviceSynchronize(...) void;
pub extern fn mpStreamSynchronize(StreamCtx) void;
pub extern fn mpInitStream(...) StreamCtx;
pub extern fn mpDeinitStream(StreamCtx) void;
pub extern fn mpCheckLastError(...) void;
pub extern fn launch_norm_l2_ij_j_r16(stream: StreamCtx, src: [*c]const r16, dst: [*c]r16, m: len_t, n: len_t) void;
pub extern fn launch_norm_l2_ij_j_r32(stream: StreamCtx, src: [*c]const r32, dst: [*c]r32, m: len_t, n: len_t) void;
pub extern fn launch_norm_l2_ij_j_r64(stream: StreamCtx, src: [*c]const r64, dst: [*c]r64, m: len_t, n: len_t) void;
pub extern fn launch_permutate_ij_ji_r16(stream: StreamCtx, src: [*c]const r16, dst: [*c]r16, dst_coef: r16, row: len_t, col: len_t) void;
pub extern fn launch_permutate_ij_ji_r32(stream: StreamCtx, src: [*c]const r32, dst: [*c]r32, dst_coef: r32, row: len_t, col: len_t) void;
pub extern fn launch_permutate_ij_ji_r64(stream: StreamCtx, src: [*c]const r64, dst: [*c]r64, dst_coef: r64, row: len_t, col: len_t) void;
pub extern fn launch_logistic_reverse_r16(stream: StreamCtx, a_grads: [*c]r16, b_value: [*c]const r16, b_grads: [*c]const r16, N: len_t) void;
pub extern fn launch_logistic_reverse_r32(stream: StreamCtx, a_grads: [*c]r32, b_value: [*c]const r32, b_grads: [*c]const r32, N: len_t) void;
pub extern fn launch_logistic_reverse_r64(stream: StreamCtx, a_grads: [*c]r64, b_value: [*c]const r64, b_grads: [*c]const r64, N: len_t) void;
pub extern fn launch_zscore_i_i_r16(stream: StreamCtx, src: [*c]const r16, dst: [*c]r16, n: len_t) void;
pub extern fn launch_zscore_i_i_r32(stream: StreamCtx, src: [*c]const r32, dst: [*c]r32, n: len_t) void;
pub extern fn launch_zscore_i_i_r64(stream: StreamCtx, src: [*c]const r64, dst: [*c]r64, n: len_t) void;
pub extern fn launch_tanh_r16(stream: StreamCtx, a: [*c]const r16, b: [*c]r16, n: len_t) void;
pub extern fn launch_tanh_r32(stream: StreamCtx, a: [*c]const r32, b: [*c]r32, n: len_t) void;
pub extern fn launch_tanh_r64(stream: StreamCtx, a: [*c]const r64, b: [*c]r64, n: len_t) void;
pub extern fn launch_extract_sort_keys_i_r16(stream: StreamCtx, pairs: [*c]const SortPair_r16, keys: [*c]c_uint, n: len_t) void;
pub extern fn launch_extract_sort_keys_i_r32(stream: StreamCtx, pairs: [*c]const SortPair_r32, keys: [*c]c_uint, n: len_t) void;
pub extern fn launch_extract_sort_keys_i_r64(stream: StreamCtx, pairs: [*c]const SortPair_r64, keys: [*c]c_uint, n: len_t) void;
pub extern fn launch_sigmoid_r16(stream: StreamCtx, a: [*c]const r16, b: [*c]r16, N: len_t) void;
pub extern fn launch_sigmoid_r32(stream: StreamCtx, a: [*c]const r32, b: [*c]r32, N: len_t) void;
pub extern fn launch_sigmoid_r64(stream: StreamCtx, a: [*c]const r64, b: [*c]r64, N: len_t) void;
pub extern fn launch_cce_loss_i_i_r16(stream: StreamCtx, src_value: [*c]const r16, src_grads: [*c]r16, trg: c_uint, scratch: [*c]r16, redux: [*c]f32, m: len_t) void;
pub extern fn launch_cce_loss_i_i_r32(stream: StreamCtx, src_value: [*c]const r32, src_grads: [*c]r32, trg: c_uint, scratch: [*c]r32, redux: [*c]f32, m: len_t) void;
pub extern fn launch_cce_loss_i_i_r64(stream: StreamCtx, src_value: [*c]const r64, src_grads: [*c]r64, trg: c_uint, scratch: [*c]r64, redux: [*c]f32, m: len_t) void;
pub extern fn launch_minmax_ij_j_r16(stream: StreamCtx, A: [*c]const r16, B: [*c]r16, m: len_t, n: len_t) void;
pub extern fn launch_minmax_ij_j_r32(stream: StreamCtx, A: [*c]const r32, B: [*c]r32, m: len_t, n: len_t) void;
pub extern fn launch_minmax_ij_j_r64(stream: StreamCtx, A: [*c]const r64, B: [*c]r64, m: len_t, n: len_t) void;
pub extern fn launch_contrastive_loss_r16(stream: StreamCtx, a_value: [*c]const r16, a_grads: [*c]r16, b_value: [*c]const r16, b_grads: [*c]r16, trg: c_uint, p_margin: f32, n_margin: f32, score: [*c]f32, n: len_t) void;
pub extern fn launch_contrastive_loss_r32(stream: StreamCtx, a_value: [*c]const r32, a_grads: [*c]r32, b_value: [*c]const r32, b_grads: [*c]r32, trg: c_uint, p_margin: f32, n_margin: f32, score: [*c]f32, n: len_t) void;
pub extern fn launch_contrastive_loss_r64(stream: StreamCtx, a_value: [*c]const r64, a_grads: [*c]r64, b_value: [*c]const r64, b_grads: [*c]r64, trg: c_uint, p_margin: f32, n_margin: f32, score: [*c]f32, n: len_t) void;
pub extern fn launch_setup_sort_pairs_i_r16(stream: StreamCtx, src: [*c]const r16, pairs: [*c]SortPair_r16, n: len_t) void;
pub extern fn launch_setup_sort_pairs_i_r32(stream: StreamCtx, src: [*c]const r32, pairs: [*c]SortPair_r32, n: len_t) void;
pub extern fn launch_setup_sort_pairs_i_r64(stream: StreamCtx, src: [*c]const r64, pairs: [*c]SortPair_r64, n: len_t) void;
pub extern fn launch_norm_l2_i_i_r16(stream: StreamCtx, src: [*c]const r16, dst: [*c]r16, n: len_t) void;
pub extern fn launch_norm_l2_i_i_r32(stream: StreamCtx, src: [*c]const r32, dst: [*c]r32, n: len_t) void;
pub extern fn launch_norm_l2_i_i_r64(stream: StreamCtx, src: [*c]const r64, dst: [*c]r64, n: len_t) void;
pub extern fn launch_hadamard_reverse_r16(stream: StreamCtx, grads_a: [*c]r16, value_b: [*c]const r16, grads_c: [*c]const r16, N: len_t) void;
pub extern fn launch_hadamard_reverse_r32(stream: StreamCtx, grads_a: [*c]r32, value_b: [*c]const r32, grads_c: [*c]const r32, N: len_t) void;
pub extern fn launch_hadamard_reverse_r64(stream: StreamCtx, grads_a: [*c]r64, value_b: [*c]const r64, grads_c: [*c]const r64, N: len_t) void;
pub extern fn launch_minmax_i_i_reverse_r16(stream: StreamCtx, src_value: [*c]const r16, src_grads: [*c]r16, dst_grads: [*c]const r16, n: len_t) void;
pub extern fn launch_minmax_i_i_reverse_r32(stream: StreamCtx, src_value: [*c]const r32, src_grads: [*c]r32, dst_grads: [*c]const r32, n: len_t) void;
pub extern fn launch_minmax_i_i_reverse_r64(stream: StreamCtx, src_value: [*c]const r64, src_grads: [*c]r64, dst_grads: [*c]const r64, n: len_t) void;
pub extern fn launch_outer_product_i_j_r16(stream: StreamCtx, x: [*c]const r16, y: [*c]const r16, alpha: r16, A: [*c]r16, beta: r16, M: len_t, N: len_t) void;
pub extern fn launch_outer_product_i_j_r32(stream: StreamCtx, x: [*c]const r32, y: [*c]const r32, alpha: r32, A: [*c]r32, beta: r32, M: len_t, N: len_t) void;
pub extern fn launch_outer_product_i_j_r64(stream: StreamCtx, x: [*c]const r64, y: [*c]const r64, alpha: r64, A: [*c]r64, beta: r64, M: len_t, N: len_t) void;
pub extern fn launch_subtraction_r16(stream: StreamCtx, a: [*c]const r16, b: [*c]const r16, c: [*c]r16, N: len_t) void;
pub extern fn launch_subtraction_r32(stream: StreamCtx, a: [*c]const r32, b: [*c]const r32, c: [*c]r32, N: len_t) void;
pub extern fn launch_subtraction_r64(stream: StreamCtx, a: [*c]const r64, b: [*c]const r64, c: [*c]r64, N: len_t) void;
pub extern fn launch_reduce_ij_i_r16(stream: StreamCtx, src: [*c]const r16, dst: [*c]r16, alpha: r16, m: len_t, n: len_t) void;
pub extern fn launch_reduce_ij_i_r32(stream: StreamCtx, src: [*c]const r32, dst: [*c]r32, alpha: r32, m: len_t, n: len_t) void;
pub extern fn launch_reduce_ij_i_r64(stream: StreamCtx, src: [*c]const r64, dst: [*c]r64, alpha: r64, m: len_t, n: len_t) void;
pub extern fn launch_linear_ij_jk_r16(stream: StreamCtx, A: [*c]const r16, B: [*c]const r16, alpha: r16, C: [*c]const r16, beta: r16, Y: [*c]r16, m: len_t, n: len_t, k: len_t) void;
pub extern fn launch_linear_ij_jk_r32(stream: StreamCtx, A: [*c]const r32, B: [*c]const r32, alpha: r32, C: [*c]const r32, beta: r32, Y: [*c]r32, m: len_t, n: len_t, k: len_t) void;
pub extern fn launch_linear_ij_jk_r64(stream: StreamCtx, A: [*c]const r64, B: [*c]const r64, alpha: r64, C: [*c]const r64, beta: r64, Y: [*c]r64, m: len_t, n: len_t, k: len_t) void;
pub extern fn launch_reduce_ij_j_r16(stream: StreamCtx, src: [*c]const r16, dst: [*c]r16, alpha: r16, m: len_t, n: len_t) void;
pub extern fn launch_reduce_ij_j_r32(stream: StreamCtx, src: [*c]const r32, dst: [*c]r32, alpha: r32, m: len_t, n: len_t) void;
pub extern fn launch_reduce_ij_j_r64(stream: StreamCtx, src: [*c]const r64, dst: [*c]r64, alpha: r64, m: len_t, n: len_t) void;
pub extern fn launch_linear_i_ij_r16(stream: StreamCtx, x: [*c]const r16, A: [*c]const r16, alpha: r16, b: [*c]const r16, beta: r16, y: [*c]r16, M: len_t, N: len_t) void;
pub extern fn launch_linear_i_ij_r32(stream: StreamCtx, x: [*c]const r32, A: [*c]const r32, alpha: r32, b: [*c]const r32, beta: r32, y: [*c]r32, M: len_t, N: len_t) void;
pub extern fn launch_linear_i_ij_r64(stream: StreamCtx, x: [*c]const r64, A: [*c]const r64, alpha: r64, b: [*c]const r64, beta: r64, y: [*c]r64, M: len_t, N: len_t) void;
pub extern fn launch_softmax_i_i_r16(stream: StreamCtx, A: [*c]const r16, B: [*c]r16, scratch: [*c]r16, m: len_t) void;
pub extern fn launch_softmax_i_i_r32(stream: StreamCtx, A: [*c]const r32, B: [*c]r32, scratch: [*c]r32, m: len_t) void;
pub extern fn launch_softmax_i_i_r64(stream: StreamCtx, A: [*c]const r64, B: [*c]r64, scratch: [*c]r64, m: len_t) void;
pub extern fn launch_convolution_2D_reverse_kernel_r16(stream: StreamCtx, src_value: [*c]const r16, kern_grads: [*c]r16, dst_grads: [*c]const r16, m: len_t, n: len_t, k_dim: len_t, windows: len_t, stride: len_t) void;
pub extern fn launch_convolution_2D_reverse_kernel_r32(stream: StreamCtx, src_value: [*c]const r32, kern_grads: [*c]r32, dst_grads: [*c]const r32, m: len_t, n: len_t, k_dim: len_t, windows: len_t, stride: len_t) void;
pub extern fn launch_convolution_2D_reverse_kernel_r64(stream: StreamCtx, src_value: [*c]const r64, kern_grads: [*c]r64, dst_grads: [*c]const r64, m: len_t, n: len_t, k_dim: len_t, windows: len_t, stride: len_t) void;
pub extern fn launch_norm_l2_ij_j_reverse_r16(stream: StreamCtx, src_value: [*c]const r16, src_grads: [*c]r16, dst_grads: [*c]const r16, m: len_t, n: len_t) void;
pub extern fn launch_norm_l2_ij_j_reverse_r32(stream: StreamCtx, src_value: [*c]const r32, src_grads: [*c]r32, dst_grads: [*c]const r32, m: len_t, n: len_t) void;
pub extern fn launch_norm_l2_ij_j_reverse_r64(stream: StreamCtx, src_value: [*c]const r64, src_grads: [*c]r64, dst_grads: [*c]const r64, m: len_t, n: len_t) void;
pub extern fn launch_selu_reverse_r16(stream: StreamCtx, a_grads: [*c]r16, b_value: [*c]const r16, b_grads: [*c]const r16, N: len_t) void;
pub extern fn launch_selu_reverse_r32(stream: StreamCtx, a_grads: [*c]r32, b_value: [*c]const r32, b_grads: [*c]const r32, N: len_t) void;
pub extern fn launch_selu_reverse_r64(stream: StreamCtx, a_grads: [*c]r64, b_value: [*c]const r64, b_grads: [*c]const r64, N: len_t) void;
pub extern fn launch_leaky_relu_r16(stream: StreamCtx, a: [*c]const r16, b: [*c]r16, coef: r16, N: len_t) void;
pub extern fn launch_leaky_relu_r32(stream: StreamCtx, a: [*c]const r32, b: [*c]r32, coef: r32, N: len_t) void;
pub extern fn launch_leaky_relu_r64(stream: StreamCtx, a: [*c]const r64, b: [*c]r64, coef: r64, N: len_t) void;
pub extern fn launch_mse_loss_i_i_r16(stream: StreamCtx, src_value: [*c]const r16, src_grads: [*c]r16, trg_value: [*c]const r16, scratch: [*c]r16, redux: [*c]f32, m: len_t) void;
pub extern fn launch_mse_loss_i_i_r32(stream: StreamCtx, src_value: [*c]const r32, src_grads: [*c]r32, trg_value: [*c]const r32, scratch: [*c]r32, redux: [*c]f32, m: len_t) void;
pub extern fn launch_mse_loss_i_i_r64(stream: StreamCtx, src_value: [*c]const r64, src_grads: [*c]r64, trg_value: [*c]const r64, scratch: [*c]r64, redux: [*c]f32, m: len_t) void;
pub extern fn launch_norm_l2_i_i_reverse_r16(stream: StreamCtx, src_value: [*c]const r16, src_grads: [*c]r16, dst_grads: [*c]const r16, n: len_t) void;
pub extern fn launch_norm_l2_i_i_reverse_r32(stream: StreamCtx, src_value: [*c]const r32, src_grads: [*c]r32, dst_grads: [*c]const r32, n: len_t) void;
pub extern fn launch_norm_l2_i_i_reverse_r64(stream: StreamCtx, src_value: [*c]const r64, src_grads: [*c]r64, dst_grads: [*c]const r64, n: len_t) void;
pub extern fn launch_reduce_key_ij_j_r16(stream: StreamCtx, src: [*c]const r16, dst: [*c]r16, keys: [*c]const c_uint, alpha: r16, scratch: [*c]r16, src_col: len_t, key_len: len_t) void;
pub extern fn launch_reduce_key_ij_j_r32(stream: StreamCtx, src: [*c]const r32, dst: [*c]r32, keys: [*c]const c_uint, alpha: r32, scratch: [*c]r32, src_col: len_t, key_len: len_t) void;
pub extern fn launch_reduce_key_ij_j_r64(stream: StreamCtx, src: [*c]const r64, dst: [*c]r64, keys: [*c]const c_uint, alpha: r64, scratch: [*c]r64, src_col: len_t, key_len: len_t) void;
pub extern fn launch_copy_indexed_ij_kj_r16(stream: StreamCtx, src: [*c]const r16, dst: [*c]r16, idxs: [*c]const len_t, src_row: len_t, src_col: len_t, out_row: len_t) void;
pub extern fn launch_copy_indexed_ij_kj_buffered_r16(stream: StreamCtx, src: [*c]const r16, dst: [*c]r16, idxs: UintBuffer, src_row: len_t, src_col: len_t, out_row: len_t) void;
pub extern fn launch_copy_indexed_ij_kj_r32(stream: StreamCtx, src: [*c]const r32, dst: [*c]r32, idxs: [*c]const len_t, src_row: len_t, src_col: len_t, out_row: len_t) void;
pub extern fn launch_copy_indexed_ij_kj_buffered_r32(stream: StreamCtx, src: [*c]const r32, dst: [*c]r32, idxs: UintBuffer, src_row: len_t, src_col: len_t, out_row: len_t) void;
pub extern fn launch_copy_indexed_ij_kj_r64(stream: StreamCtx, src: [*c]const r64, dst: [*c]r64, idxs: [*c]const len_t, src_row: len_t, src_col: len_t, out_row: len_t) void;
pub extern fn launch_copy_indexed_ij_kj_buffered_r64(stream: StreamCtx, src: [*c]const r64, dst: [*c]r64, idxs: UintBuffer, src_row: len_t, src_col: len_t, out_row: len_t) void;
pub extern fn launch_fill_r16(stream: StreamCtx, dev_a: [*c]r16, value: r16, N: len_t) void;
pub extern fn launch_fill_r32(stream: StreamCtx, dev_a: [*c]r32, value: r32, N: len_t) void;
pub extern fn launch_fill_r64(stream: StreamCtx, dev_a: [*c]r64, value: r64, N: len_t) void;
pub extern fn launch_permutate_r16(X: RTensor32, Y: RTensor32, P: Permutation) void;
pub extern fn launch_permutate_r32(X: RTensor32, Y: RTensor32, P: Permutation) void;
pub extern fn launch_permutate_r64(X: RTensor32, Y: RTensor32, P: Permutation) void;
pub extern fn launch_minmax_i_i_r16(stream: StreamCtx, src: [*c]const r16, dst: [*c]r16, n: len_t) void;
pub extern fn launch_minmax_i_i_r32(stream: StreamCtx, src: [*c]const r32, dst: [*c]r32, n: len_t) void;
pub extern fn launch_minmax_i_i_r64(stream: StreamCtx, src: [*c]const r64, dst: [*c]r64, n: len_t) void;
pub extern fn launch_broadcast_i_ij_r16(stream: StreamCtx, src: [*c]const r16, dst: [*c]r16, alpha: r16, m: len_t, n: len_t) void;
pub extern fn launch_broadcast_i_ij_r32(stream: StreamCtx, src: [*c]const r32, dst: [*c]r32, alpha: r32, m: len_t, n: len_t) void;
pub extern fn launch_broadcast_i_ij_r64(stream: StreamCtx, src: [*c]const r64, dst: [*c]r64, alpha: r64, m: len_t, n: len_t) void;
pub extern fn launch_linear_ij_kj_r16(stream: StreamCtx, A: [*c]const r16, B: [*c]const r16, alpha: r16, C: [*c]const r16, beta: r16, Y: [*c]r16, m: len_t, n: len_t, k: len_t) void;
pub extern fn launch_linear_ij_kj_r32(stream: StreamCtx, A: [*c]const r32, B: [*c]const r32, alpha: r32, C: [*c]const r32, beta: r32, Y: [*c]r32, m: len_t, n: len_t, k: len_t) void;
pub extern fn launch_linear_ij_kj_r64(stream: StreamCtx, A: [*c]const r64, B: [*c]const r64, alpha: r64, C: [*c]const r64, beta: r64, Y: [*c]r64, m: len_t, n: len_t, k: len_t) void;
pub extern fn launch_relu_leaky_reverse_r16(stream: StreamCtx, a_value: [*c]const r16, a_grads: [*c]r16, b_grads: [*c]const r16, coef: r16, N: len_t) void;
pub extern fn launch_relu_leaky_reverse_r32(stream: StreamCtx, a_value: [*c]const r32, a_grads: [*c]r32, b_grads: [*c]const r32, coef: r32, N: len_t) void;
pub extern fn launch_relu_leaky_reverse_r64(stream: StreamCtx, a_value: [*c]const r64, a_grads: [*c]r64, b_grads: [*c]const r64, coef: r64, N: len_t) void;
pub extern fn launch_addition_r16(stream: StreamCtx, a: [*c]const r16, b: [*c]const r16, c: [*c]r16, N: len_t) void;
pub extern fn launch_addition_r32(stream: StreamCtx, a: [*c]const r32, b: [*c]const r32, c: [*c]r32, N: len_t) void;
pub extern fn launch_addition_r64(stream: StreamCtx, a: [*c]const r64, b: [*c]const r64, c: [*c]r64, N: len_t) void;
pub extern fn launch_broadcast_j_ij_r16(stream: StreamCtx, src: [*c]const r16, dst: [*c]r16, alpha: r16, m: len_t, n: len_t) void;
pub extern fn launch_broadcast_j_ij_r32(stream: StreamCtx, src: [*c]const r32, dst: [*c]r32, alpha: r32, m: len_t, n: len_t) void;
pub extern fn launch_broadcast_j_ij_r64(stream: StreamCtx, src: [*c]const r64, dst: [*c]r64, alpha: r64, m: len_t, n: len_t) void;
pub extern fn launch_kernel_sort_key_i_r16(stream: StreamCtx, gpu_p1: [*c]SortPair_r16, per_thread_remaining: [*c]c_uint, n: len_t) void;
pub extern fn launch_kernel_sort_key_i_r32(stream: StreamCtx, gpu_p1: [*c]SortPair_r32, per_thread_remaining: [*c]c_uint, n: len_t) void;
pub extern fn launch_kernel_sort_key_i_r64(stream: StreamCtx, gpu_p1: [*c]SortPair_r64, per_thread_remaining: [*c]c_uint, n: len_t) void;
pub extern fn launch_momentum_r16(stream: StreamCtx, a_value: [*c]r16, a_grads: [*c]const r16, mtm: [*c]r16, rate: r16, alpha: r16, lower: r16, upper: r16, n: len_t) void;
pub extern fn launch_momentum_r32(stream: StreamCtx, a_value: [*c]r32, a_grads: [*c]const r32, mtm: [*c]r32, rate: r32, alpha: r32, lower: r32, upper: r32, n: len_t) void;
pub extern fn launch_momentum_r64(stream: StreamCtx, a_value: [*c]r64, a_grads: [*c]const r64, mtm: [*c]r64, rate: r64, alpha: r64, lower: r64, upper: r64, n: len_t) void;
pub extern fn launch_softmax_ij_j_r16(stream: StreamCtx, A: [*c]const r16, B: [*c]r16, m: len_t, n: len_t) void;
pub extern fn launch_softmax_ij_j_r32(stream: StreamCtx, A: [*c]const r32, B: [*c]r32, m: len_t, n: len_t) void;
pub extern fn launch_softmax_ij_j_r64(stream: StreamCtx, A: [*c]const r64, B: [*c]r64, m: len_t, n: len_t) void;
pub extern fn launch_bce_loss_i_i_r16(stream: StreamCtx, src_value: [*c]const r16, src_grads: [*c]r16, trg_value: [*c]const r16, scratch: [*c]r16, redux: [*c]f32, m: len_t) void;
pub extern fn launch_bce_loss_i_i_r32(stream: StreamCtx, src_value: [*c]const r32, src_grads: [*c]r32, trg_value: [*c]const r32, scratch: [*c]r32, redux: [*c]f32, m: len_t) void;
pub extern fn launch_bce_loss_i_i_r64(stream: StreamCtx, src_value: [*c]const r64, src_grads: [*c]r64, trg_value: [*c]const r64, scratch: [*c]r64, redux: [*c]f32, m: len_t) void;
pub extern fn launch_softmax_i_i_reverse_r16(stream: StreamCtx, a_grads: [*c]r16, b_value: [*c]const r16, b_grads: [*c]const r16, scratch: [*c]r16, m: len_t) void;
pub extern fn launch_softmax_i_i_reverse_r32(stream: StreamCtx, a_grads: [*c]r32, b_value: [*c]const r32, b_grads: [*c]const r32, scratch: [*c]r32, m: len_t) void;
pub extern fn launch_softmax_i_i_reverse_r64(stream: StreamCtx, a_grads: [*c]r64, b_value: [*c]const r64, b_grads: [*c]const r64, scratch: [*c]r64, m: len_t) void;
pub extern fn launch_mse_loss_ij_j_r16(stream: StreamCtx, src_value: [*c]const r16, src_grads: [*c]r16, trgs: [*c]const len_t, scratch: [*c]r16, redux: [*c]f32, m: len_t, n: len_t) void;
pub extern fn launch_mse_loss_ij_j_r32(stream: StreamCtx, src_value: [*c]const r32, src_grads: [*c]r32, trgs: [*c]const len_t, scratch: [*c]r32, redux: [*c]f32, m: len_t, n: len_t) void;
pub extern fn launch_mse_loss_ij_j_r64(stream: StreamCtx, src_value: [*c]const r64, src_grads: [*c]r64, trgs: [*c]const len_t, scratch: [*c]r64, redux: [*c]f32, m: len_t, n: len_t) void;
pub extern fn launch_addition_reverse_r16(stream: StreamCtx, a: [*c]r16, b: [*c]const r16, N: len_t) void;
pub extern fn launch_addition_reverse_r32(stream: StreamCtx, a: [*c]r32, b: [*c]const r32, N: len_t) void;
pub extern fn launch_addition_reverse_r64(stream: StreamCtx, a: [*c]r64, b: [*c]const r64, N: len_t) void;
pub extern fn launch_convolution_2D_r16(stream: StreamCtx, src: [*c]const r16, kern: [*c]const r16, dst: [*c]r16, m: len_t, n: len_t, k_dim: len_t, windows: len_t, stride: len_t) void;
pub extern fn launch_convolution_2D_r32(stream: StreamCtx, src: [*c]const r32, kern: [*c]const r32, dst: [*c]r32, m: len_t, n: len_t, k_dim: len_t, windows: len_t, stride: len_t) void;
pub extern fn launch_convolution_2D_r64(stream: StreamCtx, src: [*c]const r64, kern: [*c]const r64, dst: [*c]r64, m: len_t, n: len_t, k_dim: len_t, windows: len_t, stride: len_t) void;
pub extern fn launch_cce_loss_ij_j_r16(stream: StreamCtx, src_value: [*c]const r16, src_grads: [*c]r16, trgs: [*c]const c_uint, scratch: [*c]r16, redux: [*c]f32, m: len_t, n: len_t) void;
pub extern fn launch_cce_loss_ij_j_r32(stream: StreamCtx, src_value: [*c]const r32, src_grads: [*c]r32, trgs: [*c]const c_uint, scratch: [*c]r32, redux: [*c]f32, m: len_t, n: len_t) void;
pub extern fn launch_cce_loss_ij_j_r64(stream: StreamCtx, src_value: [*c]const r64, src_grads: [*c]r64, trgs: [*c]const c_uint, scratch: [*c]r64, redux: [*c]f32, m: len_t, n: len_t) void;
pub extern fn launch_sequence_r16(stream: StreamCtx, dev_a: [*c]r16, init: r16, step: r16, N: len_t) void;
pub extern fn launch_sequence_r32(stream: StreamCtx, dev_a: [*c]r32, init: r32, step: r32, N: len_t) void;
pub extern fn launch_sequence_r64(stream: StreamCtx, dev_a: [*c]r64, init: r64, step: r64, N: len_t) void;
pub extern fn launch_subtraction_reverse_r16(stream: StreamCtx, a: [*c]r16, b: [*c]const r16, coef: r16, N: len_t) void;
pub extern fn launch_subtraction_reverse_r32(stream: StreamCtx, a: [*c]r32, b: [*c]const r32, coef: r32, N: len_t) void;
pub extern fn launch_subtraction_reverse_r64(stream: StreamCtx, a: [*c]r64, b: [*c]const r64, coef: r64, N: len_t) void;
pub extern fn launch_gradient_descent_r16(stream: StreamCtx, a_value: [*c]r16, a_grads: [*c]const r16, rate: r16, lower: r16, upper: r16, n: len_t) void;
pub extern fn launch_gradient_descent_r32(stream: StreamCtx, a_value: [*c]r32, a_grads: [*c]const r32, rate: r32, lower: r32, upper: r32, n: len_t) void;
pub extern fn launch_gradient_descent_r64(stream: StreamCtx, a_value: [*c]r64, a_grads: [*c]const r64, rate: r64, lower: r64, upper: r64, n: len_t) void;
pub extern fn launch_minmax_ij_j_reverse_r16(stream: StreamCtx, a_value: [*c]const r16, a_grads: [*c]r16, b_grads: [*c]const r16, m: len_t, n: len_t) void;
pub extern fn launch_minmax_ij_j_reverse_r32(stream: StreamCtx, a_value: [*c]const r32, a_grads: [*c]r32, b_grads: [*c]const r32, m: len_t, n: len_t) void;
pub extern fn launch_minmax_ij_j_reverse_r64(stream: StreamCtx, a_value: [*c]const r64, a_grads: [*c]r64, b_grads: [*c]const r64, m: len_t, n: len_t) void;
pub extern fn launch_convolution_2D_source_reverse_r16(stream: StreamCtx, src_grads: [*c]r16, kern_value: [*c]const r16, dst_grads: [*c]const r16, m: len_t, n: len_t, k_dim: len_t, windows: len_t, stride: len_t) void;
pub extern fn launch_convolution_2D_source_reverse_r32(stream: StreamCtx, src_grads: [*c]r32, kern_value: [*c]const r32, dst_grads: [*c]const r32, m: len_t, n: len_t, k_dim: len_t, windows: len_t, stride: len_t) void;
pub extern fn launch_convolution_2D_source_reverse_r64(stream: StreamCtx, src_grads: [*c]r64, kern_value: [*c]const r64, dst_grads: [*c]const r64, m: len_t, n: len_t, k_dim: len_t, windows: len_t, stride: len_t) void;
pub extern fn launch_softmax_ij_j_reverse_r16(stream: StreamCtx, A_grads: [*c]r16, B_value: [*c]const r16, B_grads: [*c]const r16, m: len_t, n: len_t) void;
pub extern fn launch_softmax_ij_j_reverse_r32(stream: StreamCtx, A_grads: [*c]r32, B_value: [*c]const r32, B_grads: [*c]const r32, m: len_t, n: len_t) void;
pub extern fn launch_softmax_ij_j_reverse_r64(stream: StreamCtx, A_grads: [*c]r64, B_value: [*c]const r64, B_grads: [*c]const r64, m: len_t, n: len_t) void;
pub extern fn launch_max_key_ij_j_r16(stream: StreamCtx, src: [*c]const r16, keys: [*c]c_uint, m: len_t, n: len_t) void;
pub extern fn launch_max_key_ij_j_r32(stream: StreamCtx, src: [*c]const r32, keys: [*c]c_uint, m: len_t, n: len_t) void;
pub extern fn launch_max_key_ij_j_r64(stream: StreamCtx, src: [*c]const r64, keys: [*c]c_uint, m: len_t, n: len_t) void;
pub extern fn launch_tanh_reverse_r16(stream: StreamCtx, a_grads: [*c]r16, b_value: [*c]const r16, b_grads: [*c]const r16, N: len_t) void;
pub extern fn launch_tanh_reverse_r32(stream: StreamCtx, a_grads: [*c]r32, b_value: [*c]const r32, b_grads: [*c]const r32, N: len_t) void;
pub extern fn launch_tanh_reverse_r64(stream: StreamCtx, a_grads: [*c]r64, b_value: [*c]const r64, b_grads: [*c]const r64, N: len_t) void;
pub extern fn launch_hadamard_r16(stream: StreamCtx, a: [*c]const r16, b: [*c]const r16, c: [*c]r16, N: len_t) void;
pub extern fn launch_hadamard_r32(stream: StreamCtx, a: [*c]const r32, b: [*c]const r32, c: [*c]r32, N: len_t) void;
pub extern fn launch_hadamard_r64(stream: StreamCtx, a: [*c]const r64, b: [*c]const r64, c: [*c]r64, N: len_t) void;
pub extern fn launch_selu_r16(stream: StreamCtx, a: [*c]const r16, b: [*c]r16, n: len_t) void;
pub extern fn launch_selu_r32(stream: StreamCtx, a: [*c]const r32, b: [*c]r32, n: len_t) void;
pub extern fn launch_selu_r64(stream: StreamCtx, a: [*c]const r64, b: [*c]r64, n: len_t) void;
pub extern fn launch_linear_ij_j_r16(stream: StreamCtx, A: [*c]const r16, x: [*c]const r16, alpha: r16, b: [*c]const r16, beta: r16, y: [*c]r16, M: len_t, N: len_t) void;
pub extern fn launch_linear_ij_j_r32(stream: StreamCtx, A: [*c]const r32, x: [*c]const r32, alpha: r32, b: [*c]const r32, beta: r32, y: [*c]r32, M: len_t, N: len_t) void;
pub extern fn launch_linear_ij_j_r64(stream: StreamCtx, A: [*c]const r64, x: [*c]const r64, alpha: r64, b: [*c]const r64, beta: r64, y: [*c]r64, M: len_t, N: len_t) void;
pub extern fn launch_copy_r16(stream: StreamCtx, src: [*c]const r16, dst: [*c]r16, n: len_t) void;
pub extern fn launch_copy_r32(stream: StreamCtx, src: [*c]const r32, dst: [*c]r32, n: len_t) void;
pub extern fn launch_copy_r64(stream: StreamCtx, src: [*c]const r64, dst: [*c]r64, n: len_t) void;
pub const __INTMAX_C_SUFFIX__ = @compileError("unable to translate macro: undefined identifier `L`"); // (no file):95:9
pub const __UINTMAX_C_SUFFIX__ = @compileError("unable to translate macro: undefined identifier `UL`"); // (no file):101:9
pub const __INT64_C_SUFFIX__ = @compileError("unable to translate macro: undefined identifier `L`"); // (no file):198:9
pub const __UINT32_C_SUFFIX__ = @compileError("unable to translate macro: undefined identifier `U`"); // (no file):220:9
pub const __UINT64_C_SUFFIX__ = @compileError("unable to translate macro: undefined identifier `UL`"); // (no file):228:9
pub const __seg_gs = @compileError("unable to translate macro: undefined identifier `address_space`"); // (no file):358:9
pub const __seg_fs = @compileError("unable to translate macro: undefined identifier `address_space`"); // (no file):359:9
pub const EXTERN_C = @compileError("unable to translate C expr: unexpected token 'extern'"); // /home/andrew/ZigCode/Metaphor/src/cuda/device_utils.h:9:13
pub const __llvm__ = @as(c_int, 1);
pub const __clang__ = @as(c_int, 1);
pub const __clang_major__ = @as(c_int, 18);
pub const __clang_minor__ = @as(c_int, 1);
pub const __clang_patchlevel__ = @as(c_int, 6);
pub const __clang_version__ = "18.1.6 (https://github.com/ziglang/zig-bootstrap 32e2c2651f0b969b60b95e9174a86e09783bf4aa)";
pub const __GNUC__ = @as(c_int, 4);
pub const __GNUC_MINOR__ = @as(c_int, 2);
pub const __GNUC_PATCHLEVEL__ = @as(c_int, 1);
pub const __GXX_ABI_VERSION = @as(c_int, 1002);
pub const __ATOMIC_RELAXED = @as(c_int, 0);
pub const __ATOMIC_CONSUME = @as(c_int, 1);
pub const __ATOMIC_ACQUIRE = @as(c_int, 2);
pub const __ATOMIC_RELEASE = @as(c_int, 3);
pub const __ATOMIC_ACQ_REL = @as(c_int, 4);
pub const __ATOMIC_SEQ_CST = @as(c_int, 5);
pub const __MEMORY_SCOPE_SYSTEM = @as(c_int, 0);
pub const __MEMORY_SCOPE_DEVICE = @as(c_int, 1);
pub const __MEMORY_SCOPE_WRKGRP = @as(c_int, 2);
pub const __MEMORY_SCOPE_WVFRNT = @as(c_int, 3);
pub const __MEMORY_SCOPE_SINGLE = @as(c_int, 4);
pub const __OPENCL_MEMORY_SCOPE_WORK_ITEM = @as(c_int, 0);
pub const __OPENCL_MEMORY_SCOPE_WORK_GROUP = @as(c_int, 1);
pub const __OPENCL_MEMORY_SCOPE_DEVICE = @as(c_int, 2);
pub const __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES = @as(c_int, 3);
pub const __OPENCL_MEMORY_SCOPE_SUB_GROUP = @as(c_int, 4);
pub const __FPCLASS_SNAN = @as(c_int, 0x0001);
pub const __FPCLASS_QNAN = @as(c_int, 0x0002);
pub const __FPCLASS_NEGINF = @as(c_int, 0x0004);
pub const __FPCLASS_NEGNORMAL = @as(c_int, 0x0008);
pub const __FPCLASS_NEGSUBNORMAL = @as(c_int, 0x0010);
pub const __FPCLASS_NEGZERO = @as(c_int, 0x0020);
pub const __FPCLASS_POSZERO = @as(c_int, 0x0040);
pub const __FPCLASS_POSSUBNORMAL = @as(c_int, 0x0080);
pub const __FPCLASS_POSNORMAL = @as(c_int, 0x0100);
pub const __FPCLASS_POSINF = @as(c_int, 0x0200);
pub const __PRAGMA_REDEFINE_EXTNAME = @as(c_int, 1);
pub const __VERSION__ = "Clang 18.1.6 (https://github.com/ziglang/zig-bootstrap 32e2c2651f0b969b60b95e9174a86e09783bf4aa)";
pub const __OBJC_BOOL_IS_BOOL = @as(c_int, 0);
pub const __CONSTANT_CFSTRINGS__ = @as(c_int, 1);
pub const __clang_literal_encoding__ = "UTF-8";
pub const __clang_wide_literal_encoding__ = "UTF-32";
pub const __ORDER_LITTLE_ENDIAN__ = @as(c_int, 1234);
pub const __ORDER_BIG_ENDIAN__ = @as(c_int, 4321);
pub const __ORDER_PDP_ENDIAN__ = @as(c_int, 3412);
pub const __BYTE_ORDER__ = __ORDER_LITTLE_ENDIAN__;
pub const __LITTLE_ENDIAN__ = @as(c_int, 1);
pub const _LP64 = @as(c_int, 1);
pub const __LP64__ = @as(c_int, 1);
pub const __CHAR_BIT__ = @as(c_int, 8);
pub const __BOOL_WIDTH__ = @as(c_int, 8);
pub const __SHRT_WIDTH__ = @as(c_int, 16);
pub const __INT_WIDTH__ = @as(c_int, 32);
pub const __LONG_WIDTH__ = @as(c_int, 64);
pub const __LLONG_WIDTH__ = @as(c_int, 64);
pub const __BITINT_MAXWIDTH__ = @import("std").zig.c_translation.promoteIntLiteral(c_int, 8388608, .decimal);
pub const __SCHAR_MAX__ = @as(c_int, 127);
pub const __SHRT_MAX__ = @as(c_int, 32767);
pub const __INT_MAX__ = @import("std").zig.c_translation.promoteIntLiteral(c_int, 2147483647, .decimal);
pub const __LONG_MAX__ = @import("std").zig.c_translation.promoteIntLiteral(c_long, 9223372036854775807, .decimal);
pub const __LONG_LONG_MAX__ = @as(c_longlong, 9223372036854775807);
pub const __WCHAR_MAX__ = @import("std").zig.c_translation.promoteIntLiteral(c_int, 2147483647, .decimal);
pub const __WCHAR_WIDTH__ = @as(c_int, 32);
pub const __WINT_MAX__ = @import("std").zig.c_translation.promoteIntLiteral(c_uint, 4294967295, .decimal);
pub const __WINT_WIDTH__ = @as(c_int, 32);
pub const __INTMAX_MAX__ = @import("std").zig.c_translation.promoteIntLiteral(c_long, 9223372036854775807, .decimal);
pub const __INTMAX_WIDTH__ = @as(c_int, 64);
pub const __SIZE_MAX__ = @import("std").zig.c_translation.promoteIntLiteral(c_ulong, 18446744073709551615, .decimal);
pub const __SIZE_WIDTH__ = @as(c_int, 64);
pub const __UINTMAX_MAX__ = @import("std").zig.c_translation.promoteIntLiteral(c_ulong, 18446744073709551615, .decimal);
pub const __UINTMAX_WIDTH__ = @as(c_int, 64);
pub const __PTRDIFF_MAX__ = @import("std").zig.c_translation.promoteIntLiteral(c_long, 9223372036854775807, .decimal);
pub const __PTRDIFF_WIDTH__ = @as(c_int, 64);
pub const __INTPTR_MAX__ = @import("std").zig.c_translation.promoteIntLiteral(c_long, 9223372036854775807, .decimal);
pub const __INTPTR_WIDTH__ = @as(c_int, 64);
pub const __UINTPTR_MAX__ = @import("std").zig.c_translation.promoteIntLiteral(c_ulong, 18446744073709551615, .decimal);
pub const __UINTPTR_WIDTH__ = @as(c_int, 64);
pub const __SIZEOF_DOUBLE__ = @as(c_int, 8);
pub const __SIZEOF_FLOAT__ = @as(c_int, 4);
pub const __SIZEOF_INT__ = @as(c_int, 4);
pub const __SIZEOF_LONG__ = @as(c_int, 8);
pub const __SIZEOF_LONG_DOUBLE__ = @as(c_int, 16);
pub const __SIZEOF_LONG_LONG__ = @as(c_int, 8);
pub const __SIZEOF_POINTER__ = @as(c_int, 8);
pub const __SIZEOF_SHORT__ = @as(c_int, 2);
pub const __SIZEOF_PTRDIFF_T__ = @as(c_int, 8);
pub const __SIZEOF_SIZE_T__ = @as(c_int, 8);
pub const __SIZEOF_WCHAR_T__ = @as(c_int, 4);
pub const __SIZEOF_WINT_T__ = @as(c_int, 4);
pub const __SIZEOF_INT128__ = @as(c_int, 16);
pub const __INTMAX_TYPE__ = c_long;
pub const __INTMAX_FMTd__ = "ld";
pub const __INTMAX_FMTi__ = "li";
pub const __UINTMAX_TYPE__ = c_ulong;
pub const __UINTMAX_FMTo__ = "lo";
pub const __UINTMAX_FMTu__ = "lu";
pub const __UINTMAX_FMTx__ = "lx";
pub const __UINTMAX_FMTX__ = "lX";
pub const __PTRDIFF_TYPE__ = c_long;
pub const __PTRDIFF_FMTd__ = "ld";
pub const __PTRDIFF_FMTi__ = "li";
pub const __INTPTR_TYPE__ = c_long;
pub const __INTPTR_FMTd__ = "ld";
pub const __INTPTR_FMTi__ = "li";
pub const __SIZE_TYPE__ = c_ulong;
pub const __SIZE_FMTo__ = "lo";
pub const __SIZE_FMTu__ = "lu";
pub const __SIZE_FMTx__ = "lx";
pub const __SIZE_FMTX__ = "lX";
pub const __WCHAR_TYPE__ = c_int;
pub const __WINT_TYPE__ = c_uint;
pub const __SIG_ATOMIC_MAX__ = @import("std").zig.c_translation.promoteIntLiteral(c_int, 2147483647, .decimal);
pub const __SIG_ATOMIC_WIDTH__ = @as(c_int, 32);
pub const __CHAR16_TYPE__ = c_ushort;
pub const __CHAR32_TYPE__ = c_uint;
pub const __UINTPTR_TYPE__ = c_ulong;
pub const __UINTPTR_FMTo__ = "lo";
pub const __UINTPTR_FMTu__ = "lu";
pub const __UINTPTR_FMTx__ = "lx";
pub const __UINTPTR_FMTX__ = "lX";
pub const __FLT16_DENORM_MIN__ = @as(f16, 5.9604644775390625e-8);
pub const __FLT16_HAS_DENORM__ = @as(c_int, 1);
pub const __FLT16_DIG__ = @as(c_int, 3);
pub const __FLT16_DECIMAL_DIG__ = @as(c_int, 5);
pub const __FLT16_EPSILON__ = @as(f16, 9.765625e-4);
pub const __FLT16_HAS_INFINITY__ = @as(c_int, 1);
pub const __FLT16_HAS_QUIET_NAN__ = @as(c_int, 1);
pub const __FLT16_MANT_DIG__ = @as(c_int, 11);
pub const __FLT16_MAX_10_EXP__ = @as(c_int, 4);
pub const __FLT16_MAX_EXP__ = @as(c_int, 16);
pub const __FLT16_MAX__ = @as(f16, 6.5504e+4);
pub const __FLT16_MIN_10_EXP__ = -@as(c_int, 4);
pub const __FLT16_MIN_EXP__ = -@as(c_int, 13);
pub const __FLT16_MIN__ = @as(f16, 6.103515625e-5);
pub const __FLT_DENORM_MIN__ = @as(f32, 1.40129846e-45);
pub const __FLT_HAS_DENORM__ = @as(c_int, 1);
pub const __FLT_DIG__ = @as(c_int, 6);
pub const __FLT_DECIMAL_DIG__ = @as(c_int, 9);
pub const __FLT_EPSILON__ = @as(f32, 1.19209290e-7);
pub const __FLT_HAS_INFINITY__ = @as(c_int, 1);
pub const __FLT_HAS_QUIET_NAN__ = @as(c_int, 1);
pub const __FLT_MANT_DIG__ = @as(c_int, 24);
pub const __FLT_MAX_10_EXP__ = @as(c_int, 38);
pub const __FLT_MAX_EXP__ = @as(c_int, 128);
pub const __FLT_MAX__ = @as(f32, 3.40282347e+38);
pub const __FLT_MIN_10_EXP__ = -@as(c_int, 37);
pub const __FLT_MIN_EXP__ = -@as(c_int, 125);
pub const __FLT_MIN__ = @as(f32, 1.17549435e-38);
pub const __DBL_DENORM_MIN__ = @as(f64, 4.9406564584124654e-324);
pub const __DBL_HAS_DENORM__ = @as(c_int, 1);
pub const __DBL_DIG__ = @as(c_int, 15);
pub const __DBL_DECIMAL_DIG__ = @as(c_int, 17);
pub const __DBL_EPSILON__ = @as(f64, 2.2204460492503131e-16);
pub const __DBL_HAS_INFINITY__ = @as(c_int, 1);
pub const __DBL_HAS_QUIET_NAN__ = @as(c_int, 1);
pub const __DBL_MANT_DIG__ = @as(c_int, 53);
pub const __DBL_MAX_10_EXP__ = @as(c_int, 308);
pub const __DBL_MAX_EXP__ = @as(c_int, 1024);
pub const __DBL_MAX__ = @as(f64, 1.7976931348623157e+308);
pub const __DBL_MIN_10_EXP__ = -@as(c_int, 307);
pub const __DBL_MIN_EXP__ = -@as(c_int, 1021);
pub const __DBL_MIN__ = @as(f64, 2.2250738585072014e-308);
pub const __LDBL_DENORM_MIN__ = @as(c_longdouble, 3.64519953188247460253e-4951);
pub const __LDBL_HAS_DENORM__ = @as(c_int, 1);
pub const __LDBL_DIG__ = @as(c_int, 18);
pub const __LDBL_DECIMAL_DIG__ = @as(c_int, 21);
pub const __LDBL_EPSILON__ = @as(c_longdouble, 1.08420217248550443401e-19);
pub const __LDBL_HAS_INFINITY__ = @as(c_int, 1);
pub const __LDBL_HAS_QUIET_NAN__ = @as(c_int, 1);
pub const __LDBL_MANT_DIG__ = @as(c_int, 64);
pub const __LDBL_MAX_10_EXP__ = @as(c_int, 4932);
pub const __LDBL_MAX_EXP__ = @as(c_int, 16384);
pub const __LDBL_MAX__ = @as(c_longdouble, 1.18973149535723176502e+4932);
pub const __LDBL_MIN_10_EXP__ = -@as(c_int, 4931);
pub const __LDBL_MIN_EXP__ = -@as(c_int, 16381);
pub const __LDBL_MIN__ = @as(c_longdouble, 3.36210314311209350626e-4932);
pub const __POINTER_WIDTH__ = @as(c_int, 64);
pub const __BIGGEST_ALIGNMENT__ = @as(c_int, 16);
pub const __WINT_UNSIGNED__ = @as(c_int, 1);
pub const __INT8_TYPE__ = i8;
pub const __INT8_FMTd__ = "hhd";
pub const __INT8_FMTi__ = "hhi";
pub const __INT8_C_SUFFIX__ = "";
pub const __INT16_TYPE__ = c_short;
pub const __INT16_FMTd__ = "hd";
pub const __INT16_FMTi__ = "hi";
pub const __INT16_C_SUFFIX__ = "";
pub const __INT32_TYPE__ = c_int;
pub const __INT32_FMTd__ = "d";
pub const __INT32_FMTi__ = "i";
pub const __INT32_C_SUFFIX__ = "";
pub const __INT64_TYPE__ = c_long;
pub const __INT64_FMTd__ = "ld";
pub const __INT64_FMTi__ = "li";
pub const __UINT8_TYPE__ = u8;
pub const __UINT8_FMTo__ = "hho";
pub const __UINT8_FMTu__ = "hhu";
pub const __UINT8_FMTx__ = "hhx";
pub const __UINT8_FMTX__ = "hhX";
pub const __UINT8_C_SUFFIX__ = "";
pub const __UINT8_MAX__ = @as(c_int, 255);
pub const __INT8_MAX__ = @as(c_int, 127);
pub const __UINT16_TYPE__ = c_ushort;
pub const __UINT16_FMTo__ = "ho";
pub const __UINT16_FMTu__ = "hu";
pub const __UINT16_FMTx__ = "hx";
pub const __UINT16_FMTX__ = "hX";
pub const __UINT16_C_SUFFIX__ = "";
pub const __UINT16_MAX__ = @import("std").zig.c_translation.promoteIntLiteral(c_int, 65535, .decimal);
pub const __INT16_MAX__ = @as(c_int, 32767);
pub const __UINT32_TYPE__ = c_uint;
pub const __UINT32_FMTo__ = "o";
pub const __UINT32_FMTu__ = "u";
pub const __UINT32_FMTx__ = "x";
pub const __UINT32_FMTX__ = "X";
pub const __UINT32_MAX__ = @import("std").zig.c_translation.promoteIntLiteral(c_uint, 4294967295, .decimal);
pub const __INT32_MAX__ = @import("std").zig.c_translation.promoteIntLiteral(c_int, 2147483647, .decimal);
pub const __UINT64_TYPE__ = c_ulong;
pub const __UINT64_FMTo__ = "lo";
pub const __UINT64_FMTu__ = "lu";
pub const __UINT64_FMTx__ = "lx";
pub const __UINT64_FMTX__ = "lX";
pub const __UINT64_MAX__ = @import("std").zig.c_translation.promoteIntLiteral(c_ulong, 18446744073709551615, .decimal);
pub const __INT64_MAX__ = @import("std").zig.c_translation.promoteIntLiteral(c_long, 9223372036854775807, .decimal);
pub const __INT_LEAST8_TYPE__ = i8;
pub const __INT_LEAST8_MAX__ = @as(c_int, 127);
pub const __INT_LEAST8_WIDTH__ = @as(c_int, 8);
pub const __INT_LEAST8_FMTd__ = "hhd";
pub const __INT_LEAST8_FMTi__ = "hhi";
pub const __UINT_LEAST8_TYPE__ = u8;
pub const __UINT_LEAST8_MAX__ = @as(c_int, 255);
pub const __UINT_LEAST8_FMTo__ = "hho";
pub const __UINT_LEAST8_FMTu__ = "hhu";
pub const __UINT_LEAST8_FMTx__ = "hhx";
pub const __UINT_LEAST8_FMTX__ = "hhX";
pub const __INT_LEAST16_TYPE__ = c_short;
pub const __INT_LEAST16_MAX__ = @as(c_int, 32767);
pub const __INT_LEAST16_WIDTH__ = @as(c_int, 16);
pub const __INT_LEAST16_FMTd__ = "hd";
pub const __INT_LEAST16_FMTi__ = "hi";
pub const __UINT_LEAST16_TYPE__ = c_ushort;
pub const __UINT_LEAST16_MAX__ = @import("std").zig.c_translation.promoteIntLiteral(c_int, 65535, .decimal);
pub const __UINT_LEAST16_FMTo__ = "ho";
pub const __UINT_LEAST16_FMTu__ = "hu";
pub const __UINT_LEAST16_FMTx__ = "hx";
pub const __UINT_LEAST16_FMTX__ = "hX";
pub const __INT_LEAST32_TYPE__ = c_int;
pub const __INT_LEAST32_MAX__ = @import("std").zig.c_translation.promoteIntLiteral(c_int, 2147483647, .decimal);
pub const __INT_LEAST32_WIDTH__ = @as(c_int, 32);
pub const __INT_LEAST32_FMTd__ = "d";
pub const __INT_LEAST32_FMTi__ = "i";
pub const __UINT_LEAST32_TYPE__ = c_uint;
pub const __UINT_LEAST32_MAX__ = @import("std").zig.c_translation.promoteIntLiteral(c_uint, 4294967295, .decimal);
pub const __UINT_LEAST32_FMTo__ = "o";
pub const __UINT_LEAST32_FMTu__ = "u";
pub const __UINT_LEAST32_FMTx__ = "x";
pub const __UINT_LEAST32_FMTX__ = "X";
pub const __INT_LEAST64_TYPE__ = c_long;
pub const __INT_LEAST64_MAX__ = @import("std").zig.c_translation.promoteIntLiteral(c_long, 9223372036854775807, .decimal);
pub const __INT_LEAST64_WIDTH__ = @as(c_int, 64);
pub const __INT_LEAST64_FMTd__ = "ld";
pub const __INT_LEAST64_FMTi__ = "li";
pub const __UINT_LEAST64_TYPE__ = c_ulong;
pub const __UINT_LEAST64_MAX__ = @import("std").zig.c_translation.promoteIntLiteral(c_ulong, 18446744073709551615, .decimal);
pub const __UINT_LEAST64_FMTo__ = "lo";
pub const __UINT_LEAST64_FMTu__ = "lu";
pub const __UINT_LEAST64_FMTx__ = "lx";
pub const __UINT_LEAST64_FMTX__ = "lX";
pub const __INT_FAST8_TYPE__ = i8;
pub const __INT_FAST8_MAX__ = @as(c_int, 127);
pub const __INT_FAST8_WIDTH__ = @as(c_int, 8);
pub const __INT_FAST8_FMTd__ = "hhd";
pub const __INT_FAST8_FMTi__ = "hhi";
pub const __UINT_FAST8_TYPE__ = u8;
pub const __UINT_FAST8_MAX__ = @as(c_int, 255);
pub const __UINT_FAST8_FMTo__ = "hho";
pub const __UINT_FAST8_FMTu__ = "hhu";
pub const __UINT_FAST8_FMTx__ = "hhx";
pub const __UINT_FAST8_FMTX__ = "hhX";
pub const __INT_FAST16_TYPE__ = c_short;
pub const __INT_FAST16_MAX__ = @as(c_int, 32767);
pub const __INT_FAST16_WIDTH__ = @as(c_int, 16);
pub const __INT_FAST16_FMTd__ = "hd";
pub const __INT_FAST16_FMTi__ = "hi";
pub const __UINT_FAST16_TYPE__ = c_ushort;
pub const __UINT_FAST16_MAX__ = @import("std").zig.c_translation.promoteIntLiteral(c_int, 65535, .decimal);
pub const __UINT_FAST16_FMTo__ = "ho";
pub const __UINT_FAST16_FMTu__ = "hu";
pub const __UINT_FAST16_FMTx__ = "hx";
pub const __UINT_FAST16_FMTX__ = "hX";
pub const __INT_FAST32_TYPE__ = c_int;
pub const __INT_FAST32_MAX__ = @import("std").zig.c_translation.promoteIntLiteral(c_int, 2147483647, .decimal);
pub const __INT_FAST32_WIDTH__ = @as(c_int, 32);
pub const __INT_FAST32_FMTd__ = "d";
pub const __INT_FAST32_FMTi__ = "i";
pub const __UINT_FAST32_TYPE__ = c_uint;
pub const __UINT_FAST32_MAX__ = @import("std").zig.c_translation.promoteIntLiteral(c_uint, 4294967295, .decimal);
pub const __UINT_FAST32_FMTo__ = "o";
pub const __UINT_FAST32_FMTu__ = "u";
pub const __UINT_FAST32_FMTx__ = "x";
pub const __UINT_FAST32_FMTX__ = "X";
pub const __INT_FAST64_TYPE__ = c_long;
pub const __INT_FAST64_MAX__ = @import("std").zig.c_translation.promoteIntLiteral(c_long, 9223372036854775807, .decimal);
pub const __INT_FAST64_WIDTH__ = @as(c_int, 64);
pub const __INT_FAST64_FMTd__ = "ld";
pub const __INT_FAST64_FMTi__ = "li";
pub const __UINT_FAST64_TYPE__ = c_ulong;
pub const __UINT_FAST64_MAX__ = @import("std").zig.c_translation.promoteIntLiteral(c_ulong, 18446744073709551615, .decimal);
pub const __UINT_FAST64_FMTo__ = "lo";
pub const __UINT_FAST64_FMTu__ = "lu";
pub const __UINT_FAST64_FMTx__ = "lx";
pub const __UINT_FAST64_FMTX__ = "lX";
pub const __USER_LABEL_PREFIX__ = "";
pub const __FINITE_MATH_ONLY__ = @as(c_int, 0);
pub const __GNUC_STDC_INLINE__ = @as(c_int, 1);
pub const __GCC_ATOMIC_TEST_AND_SET_TRUEVAL = @as(c_int, 1);
pub const __CLANG_ATOMIC_BOOL_LOCK_FREE = @as(c_int, 2);
pub const __CLANG_ATOMIC_CHAR_LOCK_FREE = @as(c_int, 2);
pub const __CLANG_ATOMIC_CHAR16_T_LOCK_FREE = @as(c_int, 2);
pub const __CLANG_ATOMIC_CHAR32_T_LOCK_FREE = @as(c_int, 2);
pub const __CLANG_ATOMIC_WCHAR_T_LOCK_FREE = @as(c_int, 2);
pub const __CLANG_ATOMIC_SHORT_LOCK_FREE = @as(c_int, 2);
pub const __CLANG_ATOMIC_INT_LOCK_FREE = @as(c_int, 2);
pub const __CLANG_ATOMIC_LONG_LOCK_FREE = @as(c_int, 2);
pub const __CLANG_ATOMIC_LLONG_LOCK_FREE = @as(c_int, 2);
pub const __CLANG_ATOMIC_POINTER_LOCK_FREE = @as(c_int, 2);
pub const __GCC_ATOMIC_BOOL_LOCK_FREE = @as(c_int, 2);
pub const __GCC_ATOMIC_CHAR_LOCK_FREE = @as(c_int, 2);
pub const __GCC_ATOMIC_CHAR16_T_LOCK_FREE = @as(c_int, 2);
pub const __GCC_ATOMIC_CHAR32_T_LOCK_FREE = @as(c_int, 2);
pub const __GCC_ATOMIC_WCHAR_T_LOCK_FREE = @as(c_int, 2);
pub const __GCC_ATOMIC_SHORT_LOCK_FREE = @as(c_int, 2);
pub const __GCC_ATOMIC_INT_LOCK_FREE = @as(c_int, 2);
pub const __GCC_ATOMIC_LONG_LOCK_FREE = @as(c_int, 2);
pub const __GCC_ATOMIC_LLONG_LOCK_FREE = @as(c_int, 2);
pub const __GCC_ATOMIC_POINTER_LOCK_FREE = @as(c_int, 2);
pub const __NO_INLINE__ = @as(c_int, 1);
pub const __PIC__ = @as(c_int, 2);
pub const __pic__ = @as(c_int, 2);
pub const __FLT_RADIX__ = @as(c_int, 2);
pub const __DECIMAL_DIG__ = __LDBL_DECIMAL_DIG__;
pub const __SSP_STRONG__ = @as(c_int, 2);
pub const __ELF__ = @as(c_int, 1);
pub const __GCC_ASM_FLAG_OUTPUTS__ = @as(c_int, 1);
pub const __code_model_small__ = @as(c_int, 1);
pub const __amd64__ = @as(c_int, 1);
pub const __amd64 = @as(c_int, 1);
pub const __x86_64 = @as(c_int, 1);
pub const __x86_64__ = @as(c_int, 1);
pub const __SEG_GS = @as(c_int, 1);
pub const __SEG_FS = @as(c_int, 1);
pub const __znver4 = @as(c_int, 1);
pub const __znver4__ = @as(c_int, 1);
pub const __tune_znver4__ = @as(c_int, 1);
pub const __REGISTER_PREFIX__ = "";
pub const __NO_MATH_INLINES = @as(c_int, 1);
pub const __AES__ = @as(c_int, 1);
pub const __VAES__ = @as(c_int, 1);
pub const __PCLMUL__ = @as(c_int, 1);
pub const __VPCLMULQDQ__ = @as(c_int, 1);
pub const __LAHF_SAHF__ = @as(c_int, 1);
pub const __LZCNT__ = @as(c_int, 1);
pub const __RDRND__ = @as(c_int, 1);
pub const __FSGSBASE__ = @as(c_int, 1);
pub const __BMI__ = @as(c_int, 1);
pub const __BMI2__ = @as(c_int, 1);
pub const __POPCNT__ = @as(c_int, 1);
pub const __PRFCHW__ = @as(c_int, 1);
pub const __RDSEED__ = @as(c_int, 1);
pub const __ADX__ = @as(c_int, 1);
pub const __MWAITX__ = @as(c_int, 1);
pub const __MOVBE__ = @as(c_int, 1);
pub const __SSE4A__ = @as(c_int, 1);
pub const __FMA__ = @as(c_int, 1);
pub const __F16C__ = @as(c_int, 1);
pub const __GFNI__ = @as(c_int, 1);
pub const __EVEX512__ = @as(c_int, 1);
pub const __AVX512CD__ = @as(c_int, 1);
pub const __AVX512VPOPCNTDQ__ = @as(c_int, 1);
pub const __AVX512VNNI__ = @as(c_int, 1);
pub const __AVX512BF16__ = @as(c_int, 1);
pub const __AVX512DQ__ = @as(c_int, 1);
pub const __AVX512BITALG__ = @as(c_int, 1);
pub const __AVX512BW__ = @as(c_int, 1);
pub const __AVX512VL__ = @as(c_int, 1);
pub const __EVEX256__ = @as(c_int, 1);
pub const __AVX512VBMI__ = @as(c_int, 1);
pub const __AVX512VBMI2__ = @as(c_int, 1);
pub const __AVX512IFMA__ = @as(c_int, 1);
pub const __SHA__ = @as(c_int, 1);
pub const __FXSR__ = @as(c_int, 1);
pub const __XSAVE__ = @as(c_int, 1);
pub const __XSAVEOPT__ = @as(c_int, 1);
pub const __XSAVEC__ = @as(c_int, 1);
pub const __XSAVES__ = @as(c_int, 1);
pub const __PKU__ = @as(c_int, 1);
pub const __CLFLUSHOPT__ = @as(c_int, 1);
pub const __CLWB__ = @as(c_int, 1);
pub const __WBNOINVD__ = @as(c_int, 1);
pub const __SHSTK__ = @as(c_int, 1);
pub const __CLZERO__ = @as(c_int, 1);
pub const __RDPID__ = @as(c_int, 1);
pub const __RDPRU__ = @as(c_int, 1);
pub const __INVPCID__ = @as(c_int, 1);
pub const __CRC32__ = @as(c_int, 1);
pub const __AVX512F__ = @as(c_int, 1);
pub const __AVX2__ = @as(c_int, 1);
pub const __AVX__ = @as(c_int, 1);
pub const __SSE4_2__ = @as(c_int, 1);
pub const __SSE4_1__ = @as(c_int, 1);
pub const __SSSE3__ = @as(c_int, 1);
pub const __SSE3__ = @as(c_int, 1);
pub const __SSE2__ = @as(c_int, 1);
pub const __SSE2_MATH__ = @as(c_int, 1);
pub const __SSE__ = @as(c_int, 1);
pub const __SSE_MATH__ = @as(c_int, 1);
pub const __MMX__ = @as(c_int, 1);
pub const __GCC_HAVE_SYNC_COMPARE_AND_SWAP_1 = @as(c_int, 1);
pub const __GCC_HAVE_SYNC_COMPARE_AND_SWAP_2 = @as(c_int, 1);
pub const __GCC_HAVE_SYNC_COMPARE_AND_SWAP_4 = @as(c_int, 1);
pub const __GCC_HAVE_SYNC_COMPARE_AND_SWAP_8 = @as(c_int, 1);
pub const __GCC_HAVE_SYNC_COMPARE_AND_SWAP_16 = @as(c_int, 1);
pub const __SIZEOF_FLOAT128__ = @as(c_int, 16);
pub const unix = @as(c_int, 1);
pub const __unix = @as(c_int, 1);
pub const __unix__ = @as(c_int, 1);
pub const linux = @as(c_int, 1);
pub const __linux = @as(c_int, 1);
pub const __linux__ = @as(c_int, 1);
pub const __gnu_linux__ = @as(c_int, 1);
pub const __FLOAT128__ = @as(c_int, 1);
pub const __STDC__ = @as(c_int, 1);
pub const __STDC_HOSTED__ = @as(c_int, 1);
pub const __STDC_VERSION__ = @as(c_long, 201710);
pub const __STDC_UTF_16__ = @as(c_int, 1);
pub const __STDC_UTF_32__ = @as(c_int, 1);
pub const __GLIBC_MINOR__ = @as(c_int, 35);
pub const _DEBUG = @as(c_int, 1);
pub const __GCC_HAVE_DWARF2_CFI_ASM = @as(c_int, 1);
pub const __DEVICE_UTILS_ZIG_H__ = "";
pub const __TYPE_INDICATORS_H__ = "";
pub const __COMPLEX_TYPES_H__ = "";
pub const MAX_DIMS = @as(c_int, 6);
pub const RScalar = f32;
pub const CScalar = c32;
pub const RTensor = RTensor32;
pub const CTensor = CTensor32;
pub const SortPair_RScalar = SortPair_r32;
pub inline fn DIMPAD(M: anytype, N: anytype) @TypeOf(@import("std").zig.c_translation.MacroArithmetic.div(M + (N - @as(c_int, 1)), N)) {
    _ = &M;
    _ = &N;
    return @import("std").zig.c_translation.MacroArithmetic.div(M + (N - @as(c_int, 1)), N);
}