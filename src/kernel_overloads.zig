
const OverloadSet = @import("overloadset.zig").OverloadSet;

const decls = @import("cimport.zig").C;

pub const kernel_tanh = OverloadSet(.{
	decls.launch_tanh_r16,
	decls.launch_tanh_r32,
	decls.launch_tanh_r64,
});

pub const kernel_hadamard_reverse = OverloadSet(.{
	decls.launch_hadamard_reverse_r16,
	decls.launch_hadamard_reverse_c16,
	decls.launch_hadamard_reverse_r32,
	decls.launch_hadamard_reverse_c32,
	decls.launch_hadamard_reverse_r64,
	decls.launch_hadamard_reverse_c64,
});

pub const kernel_subtraction = OverloadSet(.{
	decls.launch_subtraction_r16,
	decls.launch_subtraction_c16,
	decls.launch_subtraction_r32,
	decls.launch_subtraction_c32,
	decls.launch_subtraction_r64,
	decls.launch_subtraction_c64,
});

pub const kernel_leaky_relu = OverloadSet(.{
	decls.launch_leaky_relu_r16,
	decls.launch_leaky_relu_r32,
	decls.launch_leaky_relu_r64,
});

pub const kernel_transpose_2D = OverloadSet(.{
	decls.launch_transpose_2D_r16,
	decls.launch_transpose_2D_r32,
	decls.launch_transpose_2D_r64,
});

pub const kernel_fill = OverloadSet(.{
	decls.launch_fill_r16,
	decls.launch_fill_c16,
	decls.launch_fill_r32,
	decls.launch_fill_c32,
	decls.launch_fill_r64,
	decls.launch_fill_c64,
});

pub const kernel_permutate = OverloadSet(.{
	decls.launch_perumutate_r16,
	decls.launch_permutate_c16,
	decls.launch_perumutate_r32,
	decls.launch_permutate_c32,
	decls.launch_perumutate_r64,
	decls.launch_permutate_c64,
});

pub const kernel_leaky_relu_reverse = OverloadSet(.{
	decls.launch_relu_leaky_reverse_r16,
	decls.launch_relu_leaky_reverse_r32,
	decls.launch_relu_leaky_reverse_r64,
});

pub const kernel_addition = OverloadSet(.{
	decls.launch_addition_r16,
	decls.launch_addition_c16,
	decls.launch_addition_r32,
	decls.launch_addition_c32,
	decls.launch_addition_r64,
	decls.launch_addition_c64,
});

pub const kernel_addition_reverse = OverloadSet(.{
	decls.launch_addition_reverse_r16,
	decls.launch_addition_reverse_c16,
	decls.launch_addition_reverse_r32,
	decls.launch_addition_reverse_c32,
	decls.launch_addition_reverse_r64,
	decls.launch_addition_reverse_c64,
});

pub const kernel_sequence = OverloadSet(.{
	decls.launch_sequence_r16,
	decls.launch_sequence_r32,
	decls.launch_sequence_r64,
});

pub const kernel_subtraction_reverse = OverloadSet(.{
	decls.launch_subtraction_reverse_r16,
	decls.launch_subtraction_reverse_c16,
	decls.launch_subtraction_reverse_r32,
	decls.launch_subtraction_reverse_c32,
	decls.launch_subtraction_reverse_r64,
	decls.launch_subtraction_reverse_c64,
});

pub const kernel_tanh_reverse = OverloadSet(.{
	decls.launch_tanh_reverse_r16,
	decls.launch_tanh_reverse_r32,
	decls.launch_tanh_reverse_r64,
});

pub const kernel_hadamard = OverloadSet(.{
	decls.launch_hadamard_r16,
	decls.launch_hadamard_c16,
	decls.launch_hadamard_r32,
	decls.launch_hadamard_c32,
	decls.launch_hadamard_r64,
	decls.launch_hadamard_c64,
});

