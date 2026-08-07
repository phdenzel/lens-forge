// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Free-form cells as `fastsum` sources — the bridge from a κ representation to the
//! [`SourceKernel`](crate::fastsum::SourceKernel)s a summation backend aggregates.

use crate::fastsum::Monopole;

/// A convergence cell reduced to its monopole moment (`strength = κ · area`) at the
/// cell centre — the [`FastSum`](crate::lens::FieldMethod::FastSum) source of a
/// free-form lens (a fastsum [`Monopole`]).
pub type MassElement = Monopole;
