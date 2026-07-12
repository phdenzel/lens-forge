// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Representation-agnostic numeric helpers shared across the crate.
//!
//! * `error_free` — error-free transformations recovering the exact round-off
//!   of a single `f64` `+`/`*`.
//! * `double_double` — `hi + lo` compensated arithmetic built on error-free
//!   transformations.

pub(crate) mod double_double;
pub(crate) mod error_free;
