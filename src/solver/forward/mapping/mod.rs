// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! The lens mapping itself: quantities derived directly from the deflection
//! field, independent of any source. Each is generic over the field
//! representation (a [`PixelGrid`](crate::grid::PixelGrid) or a
//! [`MeshGrid`](crate::grid::MeshGrid)).
//!
//! * `jacobian` — the Jacobian determinant det A = det(I − H) and its map.
//! * `magnification` — signed magnification μ(θ) = 1/det A.
//! * `convergence` — convergence κ(θ) = ½∇²ψ.
//! * `critical` — critical curves (det A = 0) and their caustics.
//! * `raytrace` — batched image-plane → source-plane ray-tracing.
//!
//! Point-source image finding (Newton polishing, the Fermat surface) lives in
//! [`extrema`](crate::solver::forward::extrema); the pixel/mesh discretizations of this map
//! (`MapDiscretization` and its realizations) live in [`basis`](crate::solver::basis).

pub mod convergence;
pub mod critical;
pub mod jacobian;
pub mod magnification;
pub mod raytrace;
