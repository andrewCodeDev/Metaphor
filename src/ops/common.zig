// standard utilities for deriving operations

pub inline fn eps_equal(x: f64, y: f64) bool {
    return @abs(x - y) < 1e-8; // or something
}
pub inline fn is_zero(x: f64) bool {
    return eps_equal(x, 0.0);
}
pub inline fn is_one(x: f64) bool {
    return eps_equal(x, 1.0);
}
