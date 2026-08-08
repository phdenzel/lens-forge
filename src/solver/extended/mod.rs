// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Extended-image world: the continuous lensed surface-brightness map.
//!
//! * `imaging` — form the lensed surface-brightness field I(θ) = S(θ − α(θ)).
//! * `operator` — the sparse lensing operator F (source → image) and its
//!   transpose, the bridge to semi-linear source inversion.

pub mod imaging;
pub mod operator;
