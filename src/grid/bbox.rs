// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Axis-aligned bounding box shared by the grid representations.

use crate::grid::Point2D;

/// An axis-aligned bounding box in the plane, given by its `[min, max]` corners.
///
/// The rectangular domain both [`Grid2D`](crate::grid::Grid2D) and the mesh
/// (`VoronoiMesh`) describe: `Grid2D` derives it from its origin and extents,
/// the mesh stores it as the clip region.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct BBox {
    /// Lower-left corner (minimum x and y).
    pub min: Point2D,
    /// Upper-right corner (maximum x and y).
    pub max: Point2D,
}

impl BBox {
    /// Create a bounding box from its lower-left and upper-right corners.
    ///
    /// # Arguments
    /// * `min` - lower-left corner.
    /// * `max` - upper-right corner.
    #[inline]
    pub const fn new(min: Point2D, max: Point2D) -> Self {
        Self { min, max }
    }

    /// Create a bounding box from per-axis limits (as `xlim`, `ylim`).
    ///
    /// # Arguments
    /// * `x_min`, `x_max` - the x extent.
    /// * `y_min`, `y_max` - the y extent.
    #[inline]
    pub const fn from_lims(x_min: f64, x_max: f64, y_min: f64, y_max: f64) -> Self {
        Self::new(Point2D::new(x_min, y_min), Point2D::new(x_max, y_max))
    }

    /// Create a bounding box from its corners as `[x, y]` arrays.
    ///
    /// # Arguments
    /// * `min` - lower-left corner as `[x, y]`.
    /// * `max` - upper-right corner as `[x, y]`.
    #[inline]
    pub const fn from_arrs(min: [f64; 2], max: [f64; 2]) -> Self {
        Self::new(Point2D::new(min[0], min[1]), Point2D::new(max[0], max[1]))
    }

    /// Whether the box is non-degenerate (`min.x < max.x && min.y < max.y`).
    #[inline]
    pub fn is_positive(&self) -> bool {
        self.min.x < self.max.x && self.min.y < self.max.y
    }

    /// Extent (width) along x.
    #[inline]
    pub fn width(&self) -> f64 {
        self.max.x - self.min.x
    }

    /// Extent (height) along y.
    #[inline]
    pub fn height(&self) -> f64 {
        self.max.y - self.min.y
    }

    /// Centre point.
    #[inline]
    pub fn center(&self) -> Point2D {
        Point2D::new(
            0.5 * (self.min.x + self.max.x),
            0.5 * (self.min.y + self.max.y),
        )
    }

    /// Enclosed area.
    #[inline]
    pub fn area(&self) -> f64 {
        self.width() * self.height()
    }

    /// Extent along x as `(min, max)`.
    #[inline]
    pub fn extent_x(&self) -> (f64, f64) {
        (self.min.x, self.max.x)
    }

    /// Extent along y as `(min, max)`.
    #[inline]
    pub fn extent_y(&self) -> (f64, f64) {
        (self.min.y, self.max.y)
    }

    /// Whether `p` lies within the closed box.
    ///
    /// # Arguments
    /// * `p` - the point to test.
    #[inline]
    pub fn contains(&self, p: &Point2D) -> bool {
        p.x >= self.min.x && p.x <= self.max.x && p.y >= self.min.y && p.y <= self.max.y
    }

    /// The four corners counter-clockwise, starting from `min`.
    #[inline]
    pub fn corners_ccw(&self) -> [Point2D; 4] {
        [
            Point2D::new(self.min.x, self.min.y),
            Point2D::new(self.max.x, self.min.y),
            Point2D::new(self.max.x, self.max.y),
            Point2D::new(self.min.x, self.max.y),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derived_quantities() {
        let b = BBox::new(Point2D::new(-1.0, -2.0), Point2D::new(3.0, 2.0));
        assert!(b.is_positive());
        assert_eq!(b.width(), 4.0);
        assert_eq!(b.height(), 4.0);
        assert_eq!(b.area(), 16.0);
        assert_eq!(b.center(), Point2D::new(1.0, 0.0));
        assert_eq!(b.extent_x(), (-1.0, 3.0));
        assert_eq!(b.extent_y(), (-2.0, 2.0));
    }

    #[test]
    fn contains_and_corners() {
        let b = BBox::new(Point2D::new(0.0, 0.0), Point2D::new(1.0, 1.0));
        assert!(b.contains(&Point2D::new(0.5, 0.5)));
        assert!(b.contains(&Point2D::new(0.0, 1.0))); // closed box
        assert!(!b.contains(&Point2D::new(1.5, 0.5)));
        assert_eq!(
            b.corners_ccw(),
            [
                Point2D::new(0.0, 0.0),
                Point2D::new(1.0, 0.0),
                Point2D::new(1.0, 1.0),
                Point2D::new(0.0, 1.0),
            ]
        );
    }

    #[test]
    fn constructors_agree() {
        let want = BBox::new(Point2D::new(-1.0, -2.0), Point2D::new(3.0, 4.0));
        assert_eq!(BBox::from_lims(-1.0, 3.0, -2.0, 4.0), want);
        assert_eq!(BBox::from_arrs([-1.0, -2.0], [3.0, 4.0]), want);
    }

    #[test]
    fn degenerate_is_not_positive() {
        assert!(!BBox::new(Point2D::new(0.0, 0.0), Point2D::new(0.0, 1.0)).is_positive());
        assert!(!BBox::new(Point2D::new(1.0, 0.0), Point2D::new(0.0, 1.0)).is_positive());
    }
}
