// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! End-to-end tests that run the compiled CLI binaries.
//!
//! These spawn the real `lens-forge`/`lf`/`lf-fits` executables (via the
//! `CARGO_BIN_EXE_*` paths) so the binary entry points and the git-style
//! dispatcher's exec/exit-code paths are exercised — the parts that in-process
//! unit tests cannot reach.

use lens_forge::grid::{Grid2D, PixelGrid};
use lens_forge::io::fits::Fits;
use std::path::{Path, PathBuf};
use std::process::Command;

/// A unique temp path for this test process and label.
fn tmp(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("lf_it_{}_{name}", std::process::id()))
}

/// Write a small valid FITS image (with a linear WCS) to `path`.
fn write_sample(path: &Path) {
    let grid = Grid2D::square(8, 2.0).unwrap();
    let pg = PixelGrid::from_func(grid, |x, y| x * x + y * y + 1.0);
    Fits::from_pixelgrid(&pg, Some(-32))
        .unwrap()
        .write(path)
        .unwrap();
}

/// The `lens-forge` dispatcher binary path.
const LENS_FORGE: &str = env!("CARGO_BIN_EXE_lens-forge");
/// The `lf` alias binary path.
const LF: &str = env!("CARGO_BIN_EXE_lf");
/// The `lf-fits` subcommand binary path.
const LF_FITS: &str = env!("CARGO_BIN_EXE_lf-fits");

#[test]
fn dispatch_runs_subcommand_and_reports_info() {
    let f = tmp("dispatch_info.fits");
    write_sample(&f);
    let out = Command::new(LENS_FORGE)
        .args(["fits", f.to_str().unwrap(), "--info"])
        .output()
        .unwrap();
    assert!(out.status.success());
    assert!(String::from_utf8_lossy(&out.stdout).contains("HDU 0"));
    let _ = std::fs::remove_file(&f);
}

#[test]
fn dispatch_propagates_child_exit_code() {
    // The subcommand fails (missing file); the dispatcher mirrors its code.
    let out = Command::new(LENS_FORGE)
        .args(["fits", "/no/such/lens_forge_file.fits"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert_eq!(out.status.code(), Some(1));
}

#[test]
fn dispatch_help_and_version() {
    for arg in ["--help", "-V"] {
        let out = Command::new(LENS_FORGE).arg(arg).output().unwrap();
        assert!(out.status.success());
        assert!(String::from_utf8_lossy(&out.stdout).contains("lens-forge"));
    }
    // No arguments prints help and succeeds.
    assert!(Command::new(LENS_FORGE).output().unwrap().status.success());
}

#[test]
fn dispatch_unknown_option_errors() {
    let out = Command::new(LENS_FORGE).arg("--bogus").output().unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("lens-forge:"));
}

#[test]
fn dispatch_unknown_subcommand_errors() {
    let out = Command::new(LENS_FORGE).arg("bogus").output().unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("available subcommands"));
}

#[test]
fn lf_alias_dispatches() {
    let f = tmp("lf_alias.fits");
    write_sample(&f);
    let out = Command::new(LF)
        .args(["fits", f.to_str().unwrap(), "--info"])
        .output()
        .unwrap();
    assert!(out.status.success());
    let _ = std::fs::remove_file(&f);
}

#[test]
fn lf_alias_unknown_option_errors() {
    let out = Command::new(LF).arg("--bogus").output().unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("lf:"));
}

#[test]
fn lf_alias_propagates_child_exit_code() {
    // Exercise the failing-child exit path through the `lf` binary too.
    let out = Command::new(LF)
        .args(["fits", "/no/such/lens_forge_file.fits"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert_eq!(out.status.code(), Some(1));
}

#[test]
fn lf_alias_unknown_subcommand_errors() {
    let out = Command::new(LF).arg("bogus").output().unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("available subcommands"));
}

#[test]
fn lf_fits_prints_header() {
    let f = tmp("lf_fits_header.fits");
    write_sample(&f);
    let out = Command::new(LF_FITS)
        .args([f.to_str().unwrap(), "--header"])
        .output()
        .unwrap();
    assert!(out.status.success());
    assert!(String::from_utf8_lossy(&out.stdout).contains("SIMPLE"));
    let _ = std::fs::remove_file(&f);
}

#[test]
fn lf_fits_saves_png() {
    let f = tmp("lf_fits_png.fits");
    write_sample(&f);
    let out_png = tmp("lf_fits_png.png");
    let out = Command::new(LF_FITS)
        .args([
            f.to_str().unwrap(),
            "-o",
            out_png.to_str().unwrap(),
            "--scale",
            "log10",
            "--cmap",
            "magma",
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(out_png.exists());
    let _ = std::fs::remove_file(&f);
    let _ = std::fs::remove_file(&out_png);
}

#[test]
fn lf_fits_missing_file_errors() {
    let out = Command::new(LF_FITS)
        .arg("/no/such/lens_forge_file.fits")
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("lf-fits:"));
}
