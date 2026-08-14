// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Command-line front end for lens-forge.
//!
//! The CLI is a git-style multi-binary tool: a `lens-forge` command (aliased
//! `lf`) dispatches to per-subcommand binaries (`lf-<name>`, e.g. `lf-fits`).
//! This module holds **only the command-line argument definitions** (clap
//! structs); the run logic lives in the `src/bin/` programs that own it.
//!
//! * [`common`] — [`CommonArgs`](common::CommonArgs), the flag set every
//!   subcommand shares (flattened into each).
//! * [`fits`] — the argument set for the `fits` subcommand.

pub mod common;
pub mod fits;
