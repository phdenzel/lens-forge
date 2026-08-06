// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! The free-form lens contract: how a κ representation becomes a lensing field.
//!
//! [`FieldMethod`] fully specifies how a [`LensJet`](crate::lens::LensJet) is
//! produced from a stored κ representation; [`FreeFormLens`] is the shared surface
//! (`build`/`new`/`with_method`/`method`/`supports`) both free-form lenses implement,
//! with the pixel-only methods rejected on an unstructured mesh via `supports`.

use crate::Result;
use crate::fastsum::{Backend, SumConfig};
use crate::lens::model::LensModel;

/// How a free-form lens turns its convergence field into a
/// [`LensJet`](crate::lens::LensJet). A value fully specifies the field.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FieldMethod {
    /// FFT Poisson solve on the pixel grid (periodic, zero-mean). Pixel-only.
    Spectral,
    /// Closed-form per-pixel log integral (free-space). Pixel-only.
    Analytic,
    /// Free-space monopole summation via a fastsum [`Backend`]. Both representations.
    FastSum {
        /// Fast-summation backend (exact direct sum, Barnes–Hut, FMM, …).
        backend: Backend,
        /// Summation configuration (softening, opening angle, order).
        config: SumConfig,
    },
}

impl FieldMethod {
    /// [`FastSum`](Self::FastSum) with the default [`SumConfig`] — the common case.
    ///
    /// # Arguments
    /// * `backend` - fast-summation backend.
    pub fn fast_sum(backend: Backend) -> Self {
        Self::FastSum {
            backend,
            config: SumConfig::default(),
        }
    }
}

/// A lens whose field is built from a κ representation via a chosen [`FieldMethod`].
///
/// The shared construction/selection surface of the free-form lenses. Not
/// object-safe (associated type, `Sized`); polymorphism stays on [`LensModel`].
pub trait FreeFormLens: LensModel + Sized {
    /// Native κ representation (`PixelGrid` / `MeshGrid`).
    type Repr;

    /// The method used when none is given (documented per lens).
    fn default_method() -> FieldMethod;

    /// Whether `method` is valid for this representation.
    ///
    /// # Arguments
    /// * `method` - field-evaluation method to test.
    fn supports(method: FieldMethod) -> bool;

    /// Build from a κ representation with an explicit `method`.
    ///
    /// # Arguments
    /// * `repr` - native κ representation.
    /// * `method` - field-evaluation method.
    ///
    /// # Errors
    /// Rejects a `method` the representation does not [`supports`](Self::supports).
    fn build(repr: Self::Repr, method: FieldMethod) -> Result<Self>;

    /// The lens's current method.
    fn method(&self) -> FieldMethod;

    /// Switch the method, rebuilding from the stored representation (chainable).
    ///
    /// # Arguments
    /// * `method` - field-evaluation method to switch to.
    ///
    /// # Errors
    /// Propagates a failed rebuild (unsupported method, or an invalid grid).
    fn with_method(self, method: FieldMethod) -> Result<Self>;

    /// Build with the [`default_method`](Self::default_method).
    ///
    /// # Arguments
    /// * `repr` - native κ representation.
    ///
    /// # Errors
    /// Propagates a failed [`build`](Self::build).
    fn new(repr: Self::Repr) -> Result<Self> {
        Self::build(repr, Self::default_method())
    }
}
