// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Composite lens: the sum of several [`LensModel`] components.
//!
//! Lensing quantities are additive in the potential, so a composite evaluates each
//! member and sums their jets (ψ, α, H) and convergence. This is where a mass profile
//! and an [`ExternalShear`](crate::lens::ExternalShear) combine into a realistic lens,
//! or where several halos overlap.

use crate::grid::Point2D;
use crate::lens::model::{LensJet, LensModel};

/// A lens formed by summing member [`LensModel`]s.
pub struct CompositeLens {
    members: Vec<Box<dyn LensModel + Send + Sync>>,
}

impl CompositeLens {
    /// Create a composite from its member lenses.
    ///
    /// # Arguments
    /// * `members` - the component lenses whose fields are summed.
    pub fn new(members: Vec<Box<dyn LensModel + Send + Sync>>) -> Self {
        Self { members }
    }

    /// Number of member lenses.
    pub fn len(&self) -> usize {
        self.members.len()
    }

    /// Whether the composite has no members (its field is then identically zero).
    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }
}

impl LensModel for CompositeLens {
    fn jet(&self, theta: Point2D) -> LensJet {
        let mut acc = LensJet::default();
        for m in &self.members {
            acc += m.jet(theta);
        }
        acc
    }

    fn jet_batch(&self, thetas: &[Point2D]) -> Vec<LensJet> {
        let mut acc = vec![LensJet::default(); thetas.len()];
        // Sum each member's batch, so free-form members keep their batched solves.
        for m in &self.members {
            for (a, j) in acc.iter_mut().zip(m.jet_batch(thetas)) {
                *a += j;
            }
        }
        acc
    }

    /// Convergence is the sum of the members' convergences.
    fn convergence(&self, theta: Point2D) -> f64 {
        self.members.iter().map(|m| m.convergence(theta)).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lens::{ExternalShear, PointMass, SIE, SIS};

    #[test]
    fn sums_member_deflections() {
        let a = PointMass::new(Point2D::new(-1.0, 0.0), 1.0);
        let b = PointMass::new(Point2D::new(1.0, 0.0), 0.8);
        let p = Point2D::new(0.3, 2.0);
        let (da, db) = (a.deflection(p), b.deflection(p));
        let comp = CompositeLens::new(vec![Box::new(a), Box::new(b)]);
        let d = comp.deflection(p);
        assert!((d.x - (da.x + db.x)).abs() < 1e-12 && (d.y - (da.y + db.y)).abs() < 1e-12);
    }

    #[test]
    fn jet_and_convergence_sum() {
        let sie = SIE::new(Point2D::origin(), 1.0, 0.7, 0.3);
        let shear = ExternalShear::new(Point2D::origin(), 0.05, -0.03);
        let p = Point2D::new(1.2, 0.6);
        let (js, jsh) = (sie.jet(p), shear.jet(p));
        let (ks, ksh) = (sie.convergence(p), shear.convergence(p));
        let comp = CompositeLens::new(vec![Box::new(sie), Box::new(shear)]);
        let j = comp.jet(p);
        assert!((j.psi - (js.psi + jsh.psi)).abs() < 1e-12);
        assert!((j.hessian[0][0] - (js.hessian[0][0] + jsh.hessian[0][0])).abs() < 1e-12);
        assert!((comp.convergence(p) - (ks + ksh)).abs() < 1e-12);
    }

    #[test]
    fn len_is_empty_and_zero_field_when_empty() {
        let empty = CompositeLens::new(vec![]);
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());
        let p = Point2D::new(1.0, 2.0);
        assert_eq!(empty.jet(p), LensJet::default()); // identically zero
        assert_eq!(empty.convergence(p), 0.0);

        let comp = CompositeLens::new(vec![Box::new(PointMass::new(Point2D::origin(), 1.0))]);
        assert_eq!(comp.len(), 1);
        assert!(!comp.is_empty());
    }

    #[test]
    fn jet_batch_matches_jet() {
        let comp = CompositeLens::new(vec![
            Box::new(SIS::new(Point2D::origin(), 1.0)),
            Box::new(ExternalShear::new(Point2D::origin(), 0.1, 0.0)),
        ]);
        let pts = [Point2D::new(1.0, 0.5), Point2D::new(-2.0, 1.0)];
        for (b, &p) in comp.jet_batch(&pts).iter().zip(&pts) {
            let s = comp.jet(p);
            assert_eq!(b.psi, s.psi);
            assert_eq!(b.alpha.x, s.alpha.x);
            assert_eq!(b.hessian[1][1], s.hessian[1][1]);
        }
    }
}
