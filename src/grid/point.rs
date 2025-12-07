// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens_forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Basic 2D point type

use std::ops::{Add, Sub, Mul, Div, Neg, AddAssign, SubAssign, MulAssign, DivAssign};


#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Point2D {
    pub x: f64,
    pub y: f64,
}


impl Point2D {
    /// Create a new point
    ///
    /// # Arguments
    /// * `x` - x coordinate
    /// * `y` - y coordinate
    #[inline]
    pub const fn new(x: f64, y: f64) -> Self {
	Self { x, y }
    }

    /// Create the origin (0, 0)
    #[inline]
    pub const fn origin() -> Self {
	Self { x: 0.0, y: 0.0 }
    }

    /// Create a unit vector along x-axis
    #[inline]
    pub const fn unit_x() -> Self {
	Self { x: 1.0, y: 0.0 }
    }

    ///  Create a unit vector along y-axis
    #[inline]
    pub const fn unit_y() -> Self {
	Self { x: 0.0, y: 1.0 }
    }

    /// Create a point with both components equal
    ///
    /// # Arguments
    /// * `v` - value for both x and y coordinates
    #[inline]
    pub const fn splat(v: f64) -> Self {
	Self { x: v, y: v }
    }

    /// Create a point from polar coordinates (r, theta)
    ///
    /// # Arguments
    /// * `r` - radial coordinate
    /// * `theta` - angular coordinate
    #[inline]
    pub fn from_polar(r: f64, theta: f64) -> Self {
        Self {
            x: r * theta.cos(),
            y: r * theta.sin(),
        }
    }

    /// Compute the Euclidean norm
    #[inline]
    pub fn norm(&self) -> f64 {
	self.norm_squared().sqrt()
    }

    /// Compute squared norm (no sqrt)
    #[inline]
    pub fn norm_squared(&self) -> f64 {
	self.x * self.x + self.y * self.y
    }

    /// Compute L1 norm (Manhattan)
    #[inline]
    pub fn norm_l1(&self) -> f64 {
	self.x.abs() + self.y.abs()
    }

    /// Compute L-inf norm (Chebyshev)
    #[inline]
    pub fn norm_linf(&self) -> f64 {
	self.x.abs().max(self.y.abs())
    }

    /// Compute Euclidean distance to another point
    ///
    /// # Arguments
    /// * `other` - reference point
    #[inline]
    pub fn distance(&self, other: &Self) -> f64 {
	(*self - *other).norm()
    }

    /// Compute squared distance to another point
    ///
    /// # Arguments
    /// * `other` - reference point
    #[inline]
    pub fn distance_squared(&self, other: &Self) -> f64 {
	(*self - *other).norm_squared()
    }

    /// Compute L1 distance to another point
    ///
    /// # Arguments
    /// * `other` - reference point
    #[inline]
    pub fn distance_l1(&self, other: &Self) -> f64 {
	(self.x - other.x).abs() + (self.y - other.y).abs()
    }

    /// Compute L-inf distance to another point
    ///
    /// # Arguments
    /// * `other` - reference point
    #[inline]
    pub fn distance_linf(&self, other: &Self) -> f64 {
	(self.x - other.x).abs().max((self.y - other.y).abs())
    }

    /// Normalize to unit length (if shorter None is returned)
    #[inline]
    pub fn normalize(&self) -> Option<Self> {
        let n = self.norm();
        if n > 1e-15 {
            Some(*self / n)
        } else {
            None
        }
    }

    /// Normalize to unit length (if shorter zero is returned)
    #[inline]
    pub fn normalize_or_zero(&self) -> Self {
        self.normalize().unwrap_or(Self::origin())
    }

    /// Normalize to a specific length
    ///
    /// # Arguments
    /// * `length` - scale factor
    #[inline]
    pub fn norm_scaled(&self, length: f64) -> Option<Self> {
        self.normalize().map(|n| n * length)
    }

    /// Linear interpolation between this point and another
    ///
    /// # Arguments
    /// * `other` - reference point
    /// * `t` - interpolation parameter
    #[inline]
    pub fn lerp(&self, other: &Self, t: f64) -> Self {
        Self {
            x: self.x + t * (other.x - self.x),
            y: self.y + t * (other.y - self.y),
        }
    }

    /// Spherical linear interpolation (for unit vectors)
    ///
    /// # Arguments
    /// * `other` - reference point
    /// * `t` - interpolation parameter
    pub fn slerp(&self, other: &Self, t: f64) -> Option<Self> {
        let dot = self.dot(other).clamp(-1.0, 1.0);
        let theta = dot.acos();
        
        if theta.abs() < 1e-10 {
            return Some(self.lerp(other, t));
        }

        let sin_theta = theta.sin();
        let a = ((1.0 - t) * theta).sin() / sin_theta;
        let b = (t * theta).sin() / sin_theta;

        Some(Self {
            x: a * self.x + b * other.x,
            y: a * self.y + b * other.y,
        })
    }

    /// Reflect this vector about a normal
    ///
    /// # Arguments
    /// * `normal` - normal vector
    #[inline]
    pub fn reflect(&self, normal: &Self) -> Self {
        *self - 2.0 * self.dot(normal) * *normal
    }

    /// Compute dot product with another point
    ///
    /// # Arguments
    /// * `other` - reference point
    #[inline]
    pub fn dot(&self, other: &Self) -> f64 {
        self.x * other.x + self.y * other.y
    }

    /// Compute 2D cross product (perp. dot product)
    ///
    /// # Arguments
    /// * `other` - reference point
    #[inline]
    pub fn cross(&self, other: &Self) -> f64 {
        self.x * other.y - self.y * other.x
    }

    /// Perpendicular vector (rotated 90 degrees counter-clockwise)
    #[inline]
    pub fn perp(&self) -> Self {
        Self { x: -self.y, y: self.x }
    }

    /// Perpendicular vector (rotated 90 degrees clockwise)
    #[inline]
    pub fn anti_perp(&self) -> Self {
        Self { x: self.y, y: -self.x }
    }

    /// Rotate by an angle (in radians) counter-clockwise
    ///
    /// # Arguments
    /// * `angle` - rotation angle
    #[inline]
    pub fn rotate(&self, angle: f64) -> Self {
        let (s, c) = angle.sin_cos();
        Self {
            x: self.x * c - self.y * s,
            y: self.x * s + self.y * c,
        }
    }

    /// Rotate around a center point
    ///
    /// # Arguments
    /// * `center` - rotation center
    /// * `angle` - rotation angle
    #[inline]
    pub fn rotate_around(&self, center: &Self, angle: f64) -> Self {
        let translated = *self - *center;
        translated.rotate(angle) + *center
    }

    /// Compute the angle (in radians [-pi, pi]) from the positive x-axis
    #[inline]
    pub fn angle(&self) -> f64 {
        self.y.atan2(self.x)
    }

    /// Compute the angle between this vector and another
    ///
    /// # Arguments
    /// * `other` - reference vector
    #[inline]
    pub fn angle_to(&self, other: &Self) -> f64 {
        let cross = self.cross(other);
        let dot = self.dot(other);
        cross.atan2(dot)
    }

    /// Convert to polar coordinates (r, theta)
    #[inline]
    pub fn to_polar(&self) -> (f64, f64) {
        (self.norm(), self.angle())
    }

    /// Project this vector onto another
    ///
    /// # Arguments
    /// * `other` - reference vector
    #[inline]
    pub fn project_onto(&self, other: &Self) -> Self {
        let other_norm_sq = other.norm_squared();
        if other_norm_sq < 1e-30 {
            Self::origin()
        } else {
            *other * (self.dot(other) / other_norm_sq)
        }
    }

    /// Reject this vector from another (perpendicular component)
    ///
    /// # Arguments
    /// * `other` - reference vector
    #[inline]
    pub fn reject_from(&self, other: &Self) -> Self {
        *self - self.project_onto(other)
    }

    /// Component-wise minimum
    ///
    /// # Arguments
    /// * `other` - reference point
    #[inline]
    pub fn min(&self, other: &Self) -> Self {
        Self {
            x: self.x.min(other.x),
            y: self.y.min(other.y),
        }
    }

    /// Component-wise maximum
    ///
    /// # Arguments
    /// * `other` - reference point
    #[inline]
    pub fn max(&self, other: &Self) -> Self {
        Self {
            x: self.x.max(other.x),
            y: self.y.max(other.y),
        }
    }

    /// Component-wise clamp
    ///
    /// # Arguments
    /// * `min` - lower bound
    /// * `max` - upper bound
    #[inline]
    pub fn clamp(&self, min: &Self, max: &Self) -> Self {
        Self {
            x: self.x.clamp(min.x, max.x),
            y: self.y.clamp(min.y, max.y),
        }
    }

    /// Component-wise absolute value
    #[inline]
    pub fn abs(&self) -> Self {
        Self {
            x: self.x.abs(),
            y: self.y.abs(),
        }
    }

    /// Component-wise floor
    #[inline]
    pub fn floor(&self) -> Self {
        Self {
            x: self.x.floor(),
            y: self.y.floor(),
        }
    }

    /// Component-wise ceil
    #[inline]
    pub fn ceil(&self) -> Self {
        Self {
            x: self.x.ceil(),
            y: self.y.ceil(),
        }
    }

    /// Component-wise round
    #[inline]
    pub fn round(&self) -> Self {
        Self {
            x: self.x.round(),
            y: self.y.round(),
        }
    }

    /// Component-wise fractional part
    #[inline]
    pub fn fract(&self) -> Self {
        Self {
            x: self.x.fract(),
            y: self.y.fract(),
        }
    }

    /// Check if any component is NaN
    #[inline]
    pub fn is_nan(&self) -> bool {
        self.x.is_nan() || self.y.is_nan()
    }

    /// Check if any component is infinite
    #[inline]
    pub fn is_infinite(&self) -> bool {
        self.x.is_infinite() || self.y.is_infinite()
    }

    /// Check if all components are finite
    #[inline]
    pub fn is_finite(&self) -> bool {
        self.x.is_finite() && self.y.is_finite()
    }

    /// Check if approximately equal to another point within epsilon
    ///
    /// # Arguments
    /// * `other` - reference point
    /// * `epsilon` - numerical precision
    #[inline]
    pub fn approx_eq(&self, other: &Self, epsilon: f64) -> bool {
        (self.x - other.x).abs() < epsilon && (self.y - other.y).abs() < epsilon
    }

    /// Check if approximately zero within epsilon
    ///
    /// # Arguments
    /// * `epsilon` - numerical precision
    #[inline]
    pub fn is_approx_zero(&self, epsilon: f64) -> bool {
        self.x.abs() < epsilon && self.y.abs() < epsilon
    }

    /// Convert to array [x, y]
    #[inline]
    pub fn to_array(&self) -> [f64; 2] {
        [self.x, self.y]
    }

    /// Create from array [x, y]
    ///
    /// # Arguments
    /// * `arr` - input array
    #[inline]
    pub fn from_array(arr: [f64; 2]) -> Self {
        Self { x: arr[0], y: arr[1] }
    }

    /// Convert to tuple (x, y)
    #[inline]
    pub fn to_tuple(&self) -> (f64, f64) {
        (self.x, self.y)
    }

    /// Create from tuple (x, y)
    ///
    /// # Arguments
    /// * `t` - input tuple
    #[inline]
    pub fn from_tuple(t: (f64, f64)) -> Self {
        Self { x: t.0, y: t.1 }
    }

    /// Compute the midpoint between this point and another
    ///
    /// # Arguments
    /// * `other` - reference point
    #[inline]
    pub fn midpoint(&self, other: &Self) -> Self {
        self.lerp(other, 0.5)
    }

    /// Compute the centroid of multiple points
    ///
    /// # Arguments
    /// * `points` - array of points
    pub fn centroid(points: &[Self]) -> Option<Self> {
        if points.is_empty() {
            return None;
        }
        let sum = points.iter().fold(Self::origin(), |acc, p| acc + *p);
        Some(sum / points.len() as f64)
    }

    /// Compute the bounding box of multiple points (min, max at corners)
    ///
    /// # Arguments
    /// * `points` - array of points
    pub fn bounding_box(points: &[Self]) -> Option<(Self, Self)> {
        if points.is_empty() {
            return None;
        }

        let mut min = points[0];
        let mut max = points[0];

        for p in points.iter().skip(1) {
            min = min.min(p);
            max = max.max(p);
        }

        Some((min, max))
    }

    /// Helper for point-in-triangle test
    ///
    /// # Arguments
    /// * `p1` - first corner of triangle
    /// * `p2` - second corner of triangle
    /// * `p3` - third corner of triangle
    fn sign(p1: &Self, p2: &Self, p3: &Self) -> f64 {
        (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y)
    }

    /// Check if this point is inside a triangle (v0, v1, v2)
    ///
    /// # Arguments
    /// * `v1` - first corner of triangle
    /// * `v2` - second corner of triangle
    /// * `v3` - third corner of triangle
    pub fn in_triangle(&self, v0: &Self, v1: &Self, v2: &Self) -> bool {
        let d1 = Self::sign(self, v0, v1);
        let d2 = Self::sign(self, v1, v2);
        let d3 = Self::sign(self, v2, v0);

        let has_neg = (d1 < 0.0) || (d2 < 0.0) || (d3 < 0.0);
        let has_pos = (d1 > 0.0) || (d2 > 0.0) || (d3 > 0.0);

        !(has_neg && has_pos)
    }

    /// Compute barycentric coordinates of this point in triangle (v0, v1, v2)
    ///
    /// # Arguments
    /// * `v1` - first corner of triangle
    /// * `v2` - second corner of triangle
    /// * `v3` - third corner of triangle
    pub fn barycentric(&self, v0: &Self, v1: &Self, v2: &Self) -> (f64, f64, f64) {
        let v0v1 = *v1 - *v0;
        let v0v2 = *v2 - *v0;
        let v0p = *self - *v0;

        let d00 = v0v1.dot(&v0v1);
        let d01 = v0v1.dot(&v0v2);
        let d11 = v0v2.dot(&v0v2);
        let d20 = v0p.dot(&v0v1);
        let d21 = v0p.dot(&v0v2);

        let denom = d00 * d11 - d01 * d01;
        if denom. abs() < 1e-30 {
            return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0);
        }

        let v = (d11 * d20 - d01 * d21) / denom;
        let w = (d00 * d21 - d01 * d20) / denom;
        let u = 1.0 - v - w;

        (u, v, w)
    }
}

/* ================ Arithmetic Operators =================================== */
/// Addition with another point (element-wise)
impl Add<Point2D> for Point2D {
    type Output = Self;
    #[inline]
    fn add(self, other: Self) -> Self {
        Self { x: self.x + other.x, y: self.y + other.y }
    }
}

/// Addition with a scalar
impl Add<f64> for Point2D {
    type Output = Self;
    #[inline]
    fn add(self, scalar: f64) -> Self {
        Self { x: self.x + scalar, y: self.y + scalar }
    }
}

/// Addition onto a scalar
impl Add<Point2D> for f64 {
    type Output = Point2D;
    #[inline]
    fn add(self, point: Point2D) -> Point2D {
        Point2D { x: self + point.x, y: self + point.y }
    }
}

/// Addition assignment with another point (element-wise)
impl AddAssign for Point2D {
    #[inline]
    fn add_assign(&mut self, other: Self) {
        self.x += other.x;
        self.y += other.y;
    }
}

/// Addition assignment with a scalar
impl AddAssign<f64> for Point2D {
    #[inline]
    fn add_assign(&mut self, scalar: f64) {
        self.x += scalar;
        self.y += scalar;
    }
}

/// Subtraction with another point (element-wise)
impl Sub<Point2D> for Point2D {
    type Output = Self;
    #[inline]
    fn sub(self, other: Self) -> Self {
        Self { x: self.x - other.x, y: self.y - other.y }
    }
}

/// Subtraction with a scalar
impl Sub<f64> for Point2D {
    type Output = Self;
    #[inline]
    fn sub(self, scalar: f64) -> Self {
        Self { x: self.x - scalar, y: self.y - scalar }
    }
}

/// Subtraction onto a scalar
impl Sub<Point2D> for f64 {
    type Output = Point2D;
    #[inline]
    fn sub(self, point: Point2D) -> Point2D {
        Point2D { x: self - point.x, y: self - point.y }
    }
}

/// Subtraction assignment with another point (element-wise)
impl SubAssign<Point2D> for Point2D {
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        self.x -= other.x;
        self.y -= other.y;
    }
}

/// Subtraction assignment with a scalar
impl SubAssign<f64> for Point2D {
    #[inline]
    fn sub_assign(&mut self, scalar: f64) {
        self.x -= scalar;
        self.y -= scalar;
    }
}

/// Multiplication with another point (element-wise)
impl Mul<Point2D> for Point2D {
    type Output = Self;
    #[inline]
    fn mul(self, other: Self) -> Self {
        Self { x: self.x * other.x, y: self.y * other.y }
    }
}

/// Multiplication with a scalar
impl Mul<f64> for Point2D {
    type Output = Self;
    #[inline]
    fn mul(self, scalar: f64) -> Self {
        Self { x: self.x * scalar, y: self.y * scalar }
    }
}

/// Multiplication onto a scalar
impl Mul<Point2D> for f64 {
    type Output = Point2D;
    #[inline]
    fn mul(self, point: Point2D) -> Point2D {
        Point2D { x: self * point.x, y: self * point.y }
    }
}

/// Multiplication assignment with another point (element-wise)
impl MulAssign<Point2D> for Point2D {
    #[inline]
    fn mul_assign(&mut self, other: Self) {
        self.x *= other.x;
        self.y *= other.y;
    }
}

/// Multiplication assignment with a scalar
impl MulAssign<f64> for Point2D {
    #[inline]
    fn mul_assign(&mut self, scalar: f64) {
        self.x *= scalar;
        self.y *= scalar;
    }
}

/// Division with another point (element-wise)
impl Div<Point2D> for Point2D {
    type Output = Self;
    #[inline]
    fn div(self, other: Self) -> Self {
        Self { x: self.x / other.x, y: self.y / other.y }
    }
}

/// Division with a scalar
impl Div<f64> for Point2D {
    type Output = Self;
    #[inline]
    fn div(self, scalar: f64) -> Self {
        Self { x: self.x / scalar, y: self.y / scalar }
    }
}

/// Division onto a scalar
impl Div<Point2D> for f64 {
    type Output = Point2D;
    #[inline]
    fn div(self, point: Point2D) -> Point2D {
        Point2D { x: self / point.x, y: self / point.y }
    }
}

/// Division assignment with another point (element-wise)
impl DivAssign<Point2D> for Point2D {
    #[inline]
    fn div_assign(&mut self, other: Self) {
        self.x /= other.x;
        self.y /= other.y;
    }
}

/// Division assignment with a scalar
impl DivAssign<f64> for Point2D {
    #[inline]
    fn div_assign(&mut self, scalar: f64) {
        self.x /= scalar;
        self.y /= scalar;
    }
}

/// Negation operator for point (element-wise)
impl Neg for Point2D {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self { x: -self.x, y: -self.y }
    }
}

/// Formatting for point
impl std::fmt::Display for Point2D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({:.6}, {:.6})", self.x, self.y)
    }
}

/// Point constructor from array
impl From<[f64; 2]> for Point2D {
    #[inline]
    fn from(arr: [f64; 2]) -> Self {
        Self::from_array(arr)
    }
}

/// Point constructor from tuple
impl From<(f64, f64)> for Point2D {
    #[inline]
    fn from(t: (f64, f64)) -> Self {
        Self::from_tuple(t)
    }
}

/// Array constructor from point
impl From<Point2D> for [f64; 2] {
    #[inline]
    fn from(p: Point2D) -> Self {
        p.to_array()
    }
}

/// Tuple constructor from point
impl From<Point2D> for (f64, f64) {
    #[inline]
    fn from(p: Point2D) -> Self {
        p.to_tuple()
    }
}


/* ================ Unit tests ============================================= */
#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{PI, FRAC_PI_2, FRAC_PI_4, SQRT_2};

    const EPSILON: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64) -> bool {
	(a - b).abs() < EPSILON
    }

    /* ============ Test constructors  ===================================== */
    #[test]
    fn test_new() {
        let p = Point2D::new(3.0, 4.0);
        assert_eq!(p.x, 3.0);
        assert_eq!(p.y, 4.0);
    }

    #[test]
    fn test_origin() {
        let p = Point2D::origin();
        assert_eq!(p.x, 0.0);
        assert_eq!(p.y, 0.0);
    }

    #[test]
    fn test_xy_unit_vectors() {
        let ux = Point2D::unit_x();
        let uy = Point2D::unit_y();

        assert_eq!(ux, Point2D::new(1.0, 0.0));
        assert_eq!(uy, Point2D::new(0.0, 1.0));
        assert!(approx_eq(ux. norm(), 1.0));
        assert!(approx_eq(uy.norm(), 1.0));
    }

    #[test]
    fn test_splat() {
        let p = Point2D::splat(5.0);
        assert_eq!(p.x, 5.0);
        assert_eq!(p.y, 5.0);
    }

    #[test]
    fn test_from_polar() {
        // r=1, theta=0 -> (1, 0)
        let p1 = Point2D::from_polar(1.0, 0.0);
        assert!(p1.approx_eq(&Point2D::new(1.0, 0.0), EPSILON));

        // r=1, theta=pi/2 -> (0, 1)
        let p2 = Point2D::from_polar(1.0, FRAC_PI_2);
        assert!(p2.approx_eq(&Point2D::new(0.0, 1.0), EPSILON));

        // r=1, theta=pi -> (-1, 0)
        let p3 = Point2D::from_polar(1.0, PI);
        assert!(p3.approx_eq(&Point2D::new(-1.0, 0.0), EPSILON));

        // r=2, theta=pi/4 -> (sqrt2, sqrt2)
        let p4 = Point2D::from_polar(2.0, FRAC_PI_4);
        assert!(p4.approx_eq(&Point2D::new(SQRT_2, SQRT_2), EPSILON));
    }

    #[test]
    fn test_to_polar() {
	// (1, 1) -> r=sqrt2, theta=pi/4
        let p = Point2D::new(1.0, 1.0);
        let (r, theta) = p.to_polar();
        assert!(approx_eq(r, SQRT_2));
        assert!(approx_eq(theta, FRAC_PI_4));
    }

    #[test]
    fn test_polar_roundtrip() {
	// (3, 4) -> (x, x) -> (3, 4)
        let original = Point2D::new(3.0, 4.0);
        let (r, theta) = original.to_polar();
        let reconstructed = Point2D::from_polar(r, theta);
        assert!(original.approx_eq(&reconstructed, EPSILON));
    }

    /* ============ Test norms & distances  ================================ */
    #[test]
    fn test_norm() {
        // 3-4-5 triangle
        let p = Point2D::new(3.0, 4.0);
        assert!(approx_eq(p.norm(), 5.0));
        assert!(approx_eq(p.norm_squared(), 25.0));
    }

    #[test]
    fn test_norm_l1() {
        let p = Point2D::new(3.0, -4.0);
        assert!(approx_eq(p.norm_l1(), 7.0));
    }

    #[test]
    fn test_norm_linf() {
        let p = Point2D::new(3.0, -5.0);
        assert!(approx_eq(p.norm_linf(), 5.0));
    }

    #[test]
    fn test_distance() {
        let p1 = Point2D::new(1.0, 2.0);
        let p2 = Point2D::new(4.0, 6.0);

        // 3-4-5 triangle
        assert!(approx_eq(p1.distance(&p2), 5.0));
        assert!(approx_eq(p1.distance_squared(&p2), 25.0));
    }

    #[test]
    fn test_distance_l1() {
        let p1 = Point2D::new(1.0, 2.0);
        let p2 = Point2D::new(4.0, 6.0);
	let res = p1.distance_l1(&p2);
	println!("{}", res);
        assert!(approx_eq(res, 7.0));
    }

    #[test]
    fn test_distance_linf() {
        let p1 = Point2D::new(1.0, 2.0);
        let p2 = Point2D::new(4.0, 6.0);
        assert!(approx_eq(p1.distance_linf(&p2), 4.0));
    }

    #[test]
    fn test_distance_symmetry() {
        let p1 = Point2D::new(1.0, 2.0);
        let p2 = Point2D::new(5.0, 7.0);
        
        assert!(approx_eq(p1.distance(&p2), p2.distance(&p1)));
        assert!(approx_eq(p1.distance_l1(&p2), p2.distance_l1(&p1)));
        assert!(approx_eq(p1.distance_linf(&p2), p2.distance_linf(&p1)));
    }

    #[test]
    fn test_normalize() {
        let p = Point2D::new(3.0, 4.0);
        let n = p.normalize().unwrap();
        
        assert!(approx_eq(n.norm(), 1.0));
        assert!(approx_eq(n.x, 0.6));
        assert!(approx_eq(n.y, 0.8));
    }

    #[test]
    fn test_normalize_zero_vector() {
        let p = Point2D::origin();
        assert!(p.normalize().is_none());
    }

    #[test]
    fn test_normalize_or_zero() {
        let p = Point2D::origin();
        let n = p. normalize_or_zero();
        assert_eq!(n, Point2D::origin());

        let p2 = Point2D::new(3.0, 4.0);
        let n2 = p2.normalize_or_zero();
        assert!(approx_eq(n2. norm(), 1.0));
    }

    #[test]
    fn test_with_norm() {
        let p = Point2D::new(3.0, 4.0);
        let scaled = p.norm_scaled(10.0).unwrap();
        
        assert!(approx_eq(scaled.norm(), 10.0));
        // Direction should be preserved
        assert!(approx_eq(scaled.x / scaled.y, p.x / p.y));
    }

    /* ============ Test polar properties ================================== */
    #[test]
    fn test_angle() {
        assert!(approx_eq(Point2D::new(1.0, 0.0).angle(), 0.0));
        assert!(approx_eq(Point2D::new(0.0, 1.0).angle(), FRAC_PI_2));
        assert!(approx_eq(Point2D::new(-1.0, 0.0).angle(), PI));
        assert!(approx_eq(Point2D::new(0.0, -1.0).angle(), -FRAC_PI_2));
        assert!(approx_eq(Point2D::new(1.0, 1.0).angle(), FRAC_PI_4));
    }

    #[test]
    fn test_angle_to() {
        let p1 = Point2D::new(1.0, 0.0);
        let p2 = Point2D::new(0.0, 1.0);
        
        assert!(approx_eq(p1.angle_to(&p2), FRAC_PI_2));
        assert!(approx_eq(p2.angle_to(&p1), -FRAC_PI_2));
    }

    /* ============ Test vectors ops ======================================= */
    #[test]
    fn test_dot() {
        let p1 = Point2D::new(2.0, 3.0);
        let p2 = Point2D::new(4.0, 5.0);
        
        // 2*4 + 3*5 = 8 + 15 = 23
        assert!(approx_eq(p1.dot(&p2), 23.0));
    }

    #[test]
    fn test_dot_perpendicular() {
        let p1 = Point2D::new(1.0, 0.0);
        let p2 = Point2D::new(0.0, 1.0);
        
        // Perpendicular vectors have zero dot product
        assert!(approx_eq(p1.dot(&p2), 0.0));
    }

    #[test]
    fn test_dot_commutative() {
        let p1 = Point2D::new(2.0, 3.0);
        let p2 = Point2D::new(4.0, 5.0);
        
        assert!(approx_eq(p1.dot(&p2), p2.dot(&p1)));
    }

    #[test]
    fn test_cross() {
        let p1 = Point2D::new(2.0, 3.0);
        let p2 = Point2D::new(4.0, 5.0);
        
        // 2*5 - 3*4 = 10 - 12 = -2
        assert!(approx_eq(p1.cross(&p2), -2.0));
    }

    #[test]
    fn test_cross_anticommutative() {
        let p1 = Point2D::new(2.0, 3.0);
        let p2 = Point2D::new(4.0, 5.0);
        
        assert!(approx_eq(p1.cross(&p2), -p2.cross(&p1)));
    }

    #[test]
    fn test_cross_parallel() {
        let p1 = Point2D::new(2.0, 3.0);
        let p2 = Point2D::new(4.0, 6.0); // p1 * 2
        
        // Parallel vectors have zero cross product
        assert!(approx_eq(p1.cross(&p2), 0.0));
    }

    #[test]
    fn test_perp() {
        let p = Point2D::new(3.0, 4.0);
        let perp = p.perp();
        
        // Should be perpendicular
        assert!(approx_eq(p.dot(&perp), 0.0));
        // Should have same length
        assert!(approx_eq(p.norm(), perp.norm()));
        // CCW rotation: (3, 4) -> (-4, 3)
        assert!(perp.approx_eq(&Point2D::new(-4.0, 3.0), EPSILON));
    }

    #[test]
    fn test_anti_perp() {
        let p = Point2D::new(3.0, 4.0);
        let perp = p.anti_perp();
        
        // Should be perpendicular
        assert!(approx_eq(p.dot(&perp), 0.0));
        // CW rotation: (3, 4) -> (4, -3)
        assert!(perp.approx_eq(&Point2D::new(4.0, -3.0), EPSILON));
    }

    #[test]
    fn test_rotate_90() {
        let p = Point2D::new(1.0, 0.0);
        let rotated = p.rotate(FRAC_PI_2);
        
        assert!(rotated.approx_eq(&Point2D::new(0.0, 1.0), EPSILON));
    }

    #[test]
    fn test_rotate_180() {
        let p = Point2D::new(3.0, 4.0);
        let rotated = p. rotate(PI);
        
        assert!(rotated.approx_eq(&Point2D::new(-3.0, -4.0), EPSILON));
    }

    #[test]
    fn test_rotate_360() {
        let p = Point2D::new(3.0, 4.0);
        let rotated = p.rotate(2.0 * PI);
        
        assert!(rotated. approx_eq(&p, EPSILON));
    }

    #[test]
    fn test_rotate_preserves_norm() {
        let p = Point2D::new(3.0, 4.0);
        let rotated = p.rotate(1.234); // arbitrary angle
        
        assert!(approx_eq(p.norm(), rotated.norm()));
    }

    #[test]
    fn test_rotate_around() {
        let p = Point2D::new(2.0, 0.0);
        let center = Point2D::new(1.0, 0.0);
        
        let rotated = p.rotate_around(&center, FRAC_PI_2);
        assert!(rotated.approx_eq(&Point2D::new(1.0, 1.0), EPSILON));
    }

    /* ============ Test point interpolation =============================== */
    #[test]
    fn test_lerp() {
        let p1 = Point2D::new(0.0, 0.0);
        let p2 = Point2D::new(10.0, 20.0);

        assert!(p1.lerp(&p2, 0.0). approx_eq(&p1, EPSILON));
        assert!(p1.lerp(&p2, 1.0).approx_eq(&p2, EPSILON));
        assert!(p1.lerp(&p2, 0.5).approx_eq(&Point2D::new(5.0, 10.0), EPSILON));
        assert!(p1.lerp(&p2, 0.25).approx_eq(&Point2D::new(2.5, 5.0), EPSILON));
    }

    #[test]
    fn test_lerp_extrapolation() {
        let p1 = Point2D::new(0.0, 0.0);
        let p2 = Point2D::new(10.0, 0.0);

        // t > 1 extrapolates beyond p2
        assert!(p1.lerp(&p2, 2.0).approx_eq(&Point2D::new(20.0, 0.0), EPSILON));
        // t < 0 extrapolates beyond p1
        assert!(p1.lerp(&p2, -1.0).approx_eq(&Point2D::new(-10.0, 0.0), EPSILON));
    }

    #[test]
    fn test_midpoint() {
        let p1 = Point2D::new(0.0, 0.0);
        let p2 = Point2D::new(10.0, 20.0);
        
        let mid = p1.midpoint(&p2);
        assert!(mid. approx_eq(&Point2D::new(5.0, 10.0), EPSILON));
    }

    #[test]
    fn test_slerp() {
        let p1 = Point2D::new(1.0, 0.0);
        let p2 = Point2D::new(0.0, 1.0);

        let mid = p1.slerp(&p2, 0.5). unwrap();
        
        // Should be on unit circle
        assert!(approx_eq(mid.norm(), 1.0));
        // Should be at 45 degrees
        assert!(approx_eq(mid.angle(), FRAC_PI_4));
    }

    /* ============ Test point projections ================================= */
    #[test]
    fn test_project_onto() {
        let v = Point2D::new(3.0, 4.0);
        let onto = Point2D::new(1.0, 0.0);
        
        let proj = v.project_onto(&onto);
        assert!(proj.approx_eq(&Point2D::new(3.0, 0.0), EPSILON));
    }

    #[test]
    fn test_reject_from() {
        let v = Point2D::new(3.0, 4.0);
        let from = Point2D::new(1.0, 0.0);
        
        let rej = v.reject_from(&from);
        assert!(rej.approx_eq(&Point2D::new(0.0, 4.0), EPSILON));
    }

    #[test]
    fn test_project_reject_sum() {
        let v = Point2D::new(3.0, 4.0);
        let onto = Point2D::new(2.0, 1.0);
        
        let proj = v.project_onto(&onto);
        let rej = v.reject_from(&onto);
        
        // proj + rej should equal original
        assert! ((proj + rej).approx_eq(&v, EPSILON));
    }

    #[test]
    fn test_reflect() {
        let v = Point2D::new(1.0, -1.0);
        let normal = Point2D::new(0.0, 1.0);
        
        let reflected = v.reflect(&normal);
        assert!(reflected.approx_eq(&Point2D::new(1.0, 1.0), EPSILON));
    }

    /* ============ Test arithmetic operations ============================= */
    #[test]
    fn test_add() {
        let p1 = Point2D::new(1.0, 2.0);
        let p2 = Point2D::new(3.0, 4.0);   
        assert_eq!(p1 + p2, Point2D::new(4.0, 6.0));
    }

    #[test]
    fn test_add_assign() {
        let mut p = Point2D::new(1.0, 2.0);
        p += Point2D::new(3.0, 4.0);
        assert_eq!(p, Point2D::new(4.0, 6.0));
    }

    #[test]
    fn test_sub() {
        let p1 = Point2D::new(5.0, 7.0);
        let p2 = Point2D::new(2.0, 3.0);
        assert_eq!(p1 - p2, Point2D::new(3.0, 4.0));
    }

    #[test]
    fn test_sub_assign() {
        let mut p = Point2D::new(5.0, 7.0);
        p -= Point2D::new(2.0, 3.0);
        assert_eq!(p, Point2D::new(3.0, 4.0));
    }

    #[test]
    fn test_mul_scalar() {
        let p = Point2D::new(2.0, 3.0);
        assert_eq!(p * 2.0, Point2D::new(4.0, 6.0));
        assert_eq!(2.0 * p, Point2D::new(4.0, 6.0));
    }

    #[test]
    fn test_componentwise_mul() {
        let p1 = Point2D::new(2.0, 3.0);
        let p2 = Point2D::new(4.0, 5.0);   
        assert_eq!(p1 * p2, Point2D::new(8.0, 15.0));
    }

    #[test]
    fn test_mul_assign() {
        let mut p = Point2D::new(2.0, 3.0);
        p *= 2.0;
        assert_eq!(p, Point2D::new(4.0, 6.0));
    }

    #[test]
    fn test_div_scalar() {
        let p = Point2D::new(4.0, 6.0);
        assert_eq!(p / 2.0, Point2D::new(2.0, 3.0));
    }

    #[test]
    fn test_componentwise_div() {
        let p1 = Point2D::new(8.0, 15.0);
        let p2 = Point2D::new(2.0, 3.0);
        assert_eq!(p1 / p2, Point2D::new(4.0, 5.0));
    }

    #[test]
    fn test_div_assign() {
        let mut p = Point2D::new(4.0, 6.0);
        p /= 2.0;
        assert_eq!(p, Point2D::new(2.0, 3.0));
    }

    #[test]
    fn test_neg() {
        let p = Point2D::new(3.0, -4.0);
        assert_eq!(-p, Point2D::new(-3.0, 4.0));
    }

    #[test]
    fn test_min_max() {
        let p1 = Point2D::new(1.0, 5.0);
        let p2 = Point2D::new(3.0, 2.0);   
        assert_eq!(p1.min(&p2), Point2D::new(1.0, 2.0));
        assert_eq!(p1.max(&p2), Point2D::new(3.0, 5.0));
    }

    #[test]
    fn test_clamp() {
        let p = Point2D::new(-1.0, 10.0);
        let min = Point2D::new(0.0, 0.0);
        let max = Point2D::new(5.0, 5.0);   
        assert_eq!(p.clamp(&min, &max), Point2D::new(0.0, 5.0));
    }

    #[test]
    fn test_abs() {
        let p = Point2D::new(-3.0, 4.0);
        assert_eq!(p.abs(), Point2D::new(3.0, 4.0));
    }

    #[test]
    fn test_floor_ceil_round() {
        let p = Point2D::new(2.7, -1.3);
        assert_eq!(p.floor(), Point2D::new(2.0, -2.0));
        assert_eq!(p.ceil(), Point2D::new(3.0, -1.0));
        assert_eq!(p.round(), Point2D::new(3.0, -1.0));
    }

    #[test]
    fn test_fract() {
        let p = Point2D::new(2.7, -1.3);
        let f = p.fract();
        assert!(approx_eq(f.x, 0.7));
        assert!(f.y.abs() < 1.0);
    }

    /* ============ Test geometric methods ================================= */
    #[test]
    fn test_centroid() {
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(3.0, 0.0),
            Point2D::new(0.0, 3.0),
        ];   
        let centroid = Point2D::centroid(&points). unwrap();
        assert!(centroid.approx_eq(&Point2D::new(1.0, 1.0), EPSILON));
    }

    #[test]
    fn test_centroid_empty() {
        let points: Vec<Point2D> = vec! [];
        assert!(Point2D::centroid(&points).is_none());
    }

    #[test]
    fn test_bounding_box() {
        let points = vec![
            Point2D::new(1.0, 2.0),
            Point2D::new(-3.0, 5.0),
            Point2D::new(4.0, -1.0),
        ];
        
        let (min, max) = Point2D::bounding_box(&points). unwrap();
        assert_eq!(min, Point2D::new(-3.0, -1.0));
        assert_eq!(max, Point2D::new(4.0, 5.0));
    }

    #[test]
    fn test_bounding_box_empty() {
        let points: Vec<Point2D> = vec![];
        assert!(Point2D::bounding_box(&points).is_none());
    }

    #[test]
    fn test_in_triangle() {
        let v0 = Point2D::new(0.0, 0.0);
        let v1 = Point2D::new(3.0, 0.0);
        let v2 = Point2D::new(0.0, 3.0);

        // Inside
        assert!(Point2D::new(0.5, 0.5).in_triangle(&v0, &v1, &v2));
        assert!(Point2D::new(1.0, 1.0).in_triangle(&v0, &v1, &v2));
        
        // Outside
        assert!(! Point2D::new(2.0, 2.0).in_triangle(&v0, &v1, &v2));
        assert!(!Point2D::new(-1.0, 0.0).in_triangle(&v0, &v1, &v2));
        
        // On edge (should return true)
        assert!(Point2D::new(1.5, 0.0).in_triangle(&v0, &v1, &v2));
    }

    #[test]
    fn test_barycentric() {
        let v0 = Point2D::new(0.0, 0.0);
        let v1 = Point2D::new(1.0, 0.0);
        let v2 = Point2D::new(0.0, 1.0);

        // At v0
        let (u, v, w) = v0.barycentric(&v0, &v1, &v2);
        assert!(approx_eq(u, 1.0) && approx_eq(v, 0.0) && approx_eq(w, 0.0));

        // At v1
        let (u, v, w) = v1.barycentric(&v0, &v1, &v2);
        assert!(approx_eq(u, 0.0) && approx_eq(v, 1.0) && approx_eq(w, 0.0));

        // At v2
        let (u, v, w) = v2. barycentric(&v0, &v1, &v2);
        assert!(approx_eq(u, 0.0) && approx_eq(v, 0.0) && approx_eq(w, 1.0));

        // Centroid
        let centroid = Point2D::new(1.0/3.0, 1.0/3.0);
        let (u, v, w) = centroid.barycentric(&v0, &v1, &v2);
        assert!(approx_eq(u, 1.0/3.0) && approx_eq(v, 1.0/3.0) && approx_eq(w, 1.0/3.0));
    }

    #[test]
    fn test_barycentric_sum_to_one() {
        let v0 = Point2D::new(0.0, 0.0);
        let v1 = Point2D::new(3.0, 0.0);
        let v2 = Point2D::new(1.0, 2.0);

        let p = Point2D::new(1.0, 0.5);
        let (u, v, w) = p.barycentric(&v0, &v1, &v2);
        
        assert!(approx_eq(u + v + w, 1.0));
    }

    /* ============ Test validation checks ================================= */
    #[test]
    fn test_is_nan() {
        assert! (!Point2D::new(1.0, 2.0). is_nan());
        assert!(Point2D::new(f64::NAN, 2.0).is_nan());
        assert!(Point2D::new(1.0, f64::NAN).is_nan());
    }

    #[test]
    fn test_is_infinite() {
        assert!(!Point2D::new(1.0, 2.0).is_infinite());
        assert!(Point2D::new(f64::INFINITY, 2.0).is_infinite());
        assert!(Point2D::new(1.0, f64::NEG_INFINITY).is_infinite());
    }

    #[test]
    fn test_is_finite() {
        assert!(Point2D::new(1.0, 2.0).is_finite());
        assert!(!Point2D::new(f64::INFINITY, 2.0).is_finite());
        assert!(!Point2D::new(f64::NAN, 2.0).is_finite());
    }

    #[test]
    fn test_approx_eq() {
        let p1 = Point2D::new(1.0, 2.0);
        let p2 = Point2D::new(1.0 + 1e-12, 2.0 - 1e-12);
        
        assert!(p1.approx_eq(&p2, 1e-10));
        assert!(!p1.approx_eq(&Point2D::new(1.1, 2.0), 1e-10));
    }

    #[test]
    fn test_is_approx_zero() {
        assert!(Point2D::new(1e-12, -1e-12).is_approx_zero(1e-10));
        assert!(!Point2D::new(0.1, 0.0).is_approx_zero(1e-10));
    }

    /* ============ Test conversions ======================================= */
    #[test]
    fn test_to_from_array() {
        let p = Point2D::new(3.0, 4.0);
        let arr = p.to_array();   
        assert_eq!(arr, [3.0, 4.0]);
        assert_eq!(Point2D::from_array(arr), p);
    }

    #[test]
    fn test_to_from_tuple() {
        let p = Point2D::new(3.0, 4.0);
        let tup = p.to_tuple();
        assert_eq!(tup, (3.0, 4.0));
        assert_eq!(Point2D::from_tuple(tup), p);
    }

    #[test]
    fn test_from_traits() {
        let p1: Point2D = [3.0, 4.0].into();
        let p2: Point2D = (3.0, 4.0).into();
        assert_eq!(p1, Point2D::new(3.0, 4.0));
        assert_eq!(p2, Point2D::new(3.0, 4.0));
    }

    #[test]
    fn test_into_traits() {
        let p = Point2D::new(3.0, 4.0);
        let arr: [f64; 2] = p.into();
        let tup: (f64, f64) = p.into();
        assert_eq!(arr, [3.0, 4.0]);
        assert_eq!(tup, (3.0, 4.0));
    }

    /* ============ Test display =========================================== */
    #[test]
    fn test_display() {
        let p = Point2D::new(1.5, -2.5);
        let s = format!("{}", p);   
        assert!(s.contains("1.5"));
        assert!(s.contains("-2.5"));
    }

    #[test]
    fn test_debug() {
        let p = Point2D::new(1.0, 2.0);
        let s = format!("{:?}", p);
        
        assert!(s.contains("Point2D"));
        assert!(s.contains("1.0"));
        assert!(s.contains("2.0"));
    }

    /* ============ Test default & clone =================================== */
    #[test]
    fn test_default() {
        let p: Point2D = Default::default();
        assert_eq!(p, Point2D::origin());
    }

    #[test]
    fn test_clone() {
        let p1 = Point2D::new(3.0, 4.0);
        let p2 = p1.clone();   
        assert_eq!(p1, p2);
    }

    #[test]
    fn test_copy() {
        let p1 = Point2D::new(3.0, 4.0);
        let p2 = p1; // Copy
        assert_eq!(p1, p2);
    }

    /* ============ Test edge cases ======================================== */
    #[test]
    fn test_very_small_numbers() {
        let p = Point2D::new(1e-152, 1e-152);
        assert!(p.is_finite());
        assert!(p.norm() > 0.0);
    }

    #[test]
    fn test_very_large_numbers() {
        let p = Point2D::new(1e152, 1e152);
        assert!(p.is_finite());
        assert!(p.norm(). is_finite());
    }

    #[test]
    fn test_mixed_signs() {
        let p = Point2D::new(-3.0, 4.0);   
        assert!(approx_eq(p.norm(), 5.0));
        assert!(approx_eq(p.norm_l1(), 7.0));
    }
}
