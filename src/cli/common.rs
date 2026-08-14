// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Arguments shared by every subcommand.
//!
//! [`CommonArgs`] is flattened into each subcommand's parser so the verbosity
//! flags are accepted uniformly. It carries no subcommand-specific state.

use clap::Args;

/// Flags common to all lens-forge subcommands.
#[derive(Args, Debug, Clone, Default)]
pub struct CommonArgs {
    /// Increase diagnostic output (repeatable: `-v`, `-vv`).
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    pub verbose: u8,
    /// Suppress non-error output on stdout.
    #[arg(short, long, global = true)]
    pub quiet: bool,
}

impl CommonArgs {
    /// The effective verbosity level (`0` when `--quiet`, else the `-v` count).
    pub fn verbosity(&self) -> u8 {
        if self.quiet { 0 } else { self.verbose }
    }

    /// Whether ordinary (non-error) output should be printed.
    pub fn is_loud(&self) -> bool {
        !self.quiet
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quiet_overrides_verbose() {
        let a = CommonArgs {
            verbose: 2,
            quiet: true,
        };
        assert_eq!(a.verbosity(), 0);
        assert!(!a.is_loud());

        let b = CommonArgs {
            verbose: 2,
            quiet: false,
        };
        assert_eq!(b.verbosity(), 2);
        assert!(b.is_loud());
    }
}
