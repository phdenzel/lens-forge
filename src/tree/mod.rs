// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Shared spatial data structures.
//!
//! * `kdtree` — balanced 2D k-d tree for nearest-site point location.

mod kdtree;
pub(crate) use kdtree::KdTree;
