// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! The `lf` command: a short alias for `lens-forge` (same dispatcher).

#[path = "../dispatch.rs"]
mod dispatch;

fn main() {
    if let Err(e) = dispatch::run() {
        eprintln!("lf: {e}");
        std::process::exit(1);
    }
}
