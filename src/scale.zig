
const std = @import("std");
const meta = std.meta;
const scl = @import("scalar.zig");
const ut = @import("utility.zig");

pub const MaxMod = struct {

    pub fn apply(x: anytype, y: anytype) scl.DemoteComplex(meta.Child(@TypeOf(x))) {
        const T = scl.DemoteComplex(std.meta.Child(@TypeOf(x)));
        var max_m: T = 0;
        var i: usize = 0;
        while (i < x.len) : (i += 1) {
            max_m = @max(max_m, scl.mod(x[i]));
        }
        i = 0;
        while (i < x.len) : (i += 1) {
            y[i] = scl.div(x[i], max_m);
        }
        return 1.0 / max_m;
    }

    pub fn reverseArg0(X: anytype, coef: anytype, Z: anytype) void {
        const x_grads = ut.assertGrads(X);
        const z_grads = ut.assertGrads(Z);
        for (0..x_grads.len) |i| { 
            x_grads[i] = scl.add(x_grads[i], scl.mul(coef, z_grads[i])); 
        }
    }
};

