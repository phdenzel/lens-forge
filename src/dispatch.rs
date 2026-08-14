// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! The `lens-forge`/`lf` dispatcher run logic.
//!
//! This file is not part of the library; it is `#[path]`-included by both the
//! `lens-forge` (`src/main.rs`) and `lf` (`src/bin/lf.rs`) binaries, which share
//! the same behaviour. Git-style external subcommands: `lf <name> <args...>`
//! runs the sibling binary `lf-<name>`, forwarding the remaining arguments and
//! propagating its exit code. The dispatcher itself only understands
//! `-h/--help` and `-V/--version`; every other flag is passed through to the
//! subcommand.

use lens_forge::{LensForgeError, Result};
use std::path::PathBuf;
use std::process::Command;

/// Subcommands advertised in the help text.
const KNOWN_SUBCOMMANDS: &[&str] = &["fits"];

/// The binary name implementing subcommand `sub` (`fits` → `lf-fits`).
///
/// # Arguments
/// * `sub` - the subcommand name.
pub fn subcommand_binary_name(sub: &str) -> String {
    format!("lf-{sub}")
}

/// Resolve the executable for subcommand `sub`, preferring a sibling of the
/// running dispatcher, then falling back to a bare name for `PATH` lookup.
fn resolve_subcommand_exe(sub: &str) -> PathBuf {
    let file = format!("{}{}", subcommand_binary_name(sub), std::env::consts::EXE_SUFFIX);
    std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|dir| dir.join(&file)))
        .filter(|sibling| sibling.is_file())
        .unwrap_or_else(|| PathBuf::from(subcommand_binary_name(sub)))
}

/// The dispatcher entry point used by the `lens-forge` and `lf` binaries.
pub fn run() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    dispatch(&args)
}

/// Dispatch already-collected arguments (the part after the program name).
///
/// # Arguments
/// * `args` - the arguments following the `lens-forge`/`lf` program name.
fn dispatch(args: &[String]) -> Result<()> {
    let Some(first) = args.first() else {
        print_help();
        return Ok(());
    };
    match first.as_str() {
        "-h" | "--help" | "help" => {
            print_help();
            return Ok(());
        }
        "-V" | "--version" => {
            println!("lens-forge {}", env!("CARGO_PKG_VERSION"));
            return Ok(());
        }
        s if s.starts_with('-') => {
            return Err(LensForgeError::invalid_option(format!(
                "unknown option {s:?}; run `lens-forge --help`"
            )));
        }
        _ => {}
    }

    let sub = &args[0];
    let rest = &args[1..];
    let exe = resolve_subcommand_exe(sub);
    let status = Command::new(&exe).args(rest).status().map_err(|e| {
        LensForgeError::io_error(format!(
            "failed to run subcommand {sub:?} (looked for {:?}): {e}; \
             available subcommands: {}",
            subcommand_binary_name(sub),
            KNOWN_SUBCOMMANDS.join(", ")
        ))
    })?;
    // Mirror the child's exit code; a plain Ok keeps 0.
    match status.code() {
        Some(0) | None => Ok(()),
        Some(code) => std::process::exit(code),
    }
}

/// Print the top-level usage and the known subcommands.
fn print_help() {
    println!("lens-forge {} — gravitational-lens modelling", env!("CARGO_PKG_VERSION"));
    println!();
    println!("Usage: lens-forge <SUBCOMMAND> [ARGS...]   (also available as `lf`)");
    println!();
    println!("Subcommands:");
    for s in KNOWN_SUBCOMMANDS {
        println!("  {s:<8} run `lf {s} --help` for its options");
    }
    println!();
    println!("Options:");
    println!("  -h, --help     Show this help");
    println!("  -V, --version  Show the version");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_name_prefixes_lf() {
        assert_eq!(subcommand_binary_name("fits"), "lf-fits");
        assert_eq!(subcommand_binary_name("solve"), "lf-solve");
    }

    #[test]
    fn help_and_version_do_not_error() {
        assert!(dispatch(&[]).is_ok());
        assert!(dispatch(&["--help".to_string()]).is_ok());
        assert!(dispatch(&["-V".to_string()]).is_ok());
    }

    #[test]
    fn unknown_option_errors() {
        assert!(dispatch(&["--bogus".to_string()]).is_err());
    }

    #[test]
    fn missing_subcommand_binary_errors() {
        // A subcommand with no matching binary anywhere yields a helpful error.
        let err = dispatch(&["definitely-not-a-subcommand".to_string()]).unwrap_err();
        assert!(matches!(err, LensForgeError::IoError(_)));
    }
}
