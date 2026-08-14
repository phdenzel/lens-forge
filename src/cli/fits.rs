// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! Command-line arguments for the `fits` subcommand.
//!
//! Only the argument definitions live here; the run logic (reading, formatting,
//! plotting) is implemented in the `lf-fits` binary that owns this command.

use clap::{Args, ValueEnum};
use std::path::PathBuf;

/// Value scaling applied before colour mapping.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default, ValueEnum)]
pub enum ScaleArg {
    /// No scaling.
    #[default]
    Linear,
    /// Base-10 logarithm.
    Log10,
    /// Symmetric log with a linear region around zero (see `--linthresh`).
    Symlog,
}

/// Arguments for the `fits` subcommand.
#[derive(Args, Debug, Clone)]
pub struct FitsArgs {
    /// Path to the FITS file.
    pub path: PathBuf,
    /// HDU index to act on.
    #[arg(long, default_value_t = 0)]
    pub hdu: usize,
    /// Print the selected HDU's header cards verbatim.
    #[arg(short = 'H', long)]
    pub header: bool,
    /// Print a whole-file metadata summary (the default action).
    #[arg(short, long)]
    pub info: bool,
    /// Render and display the image in an interactive window.
    #[arg(long)]
    pub plot: bool,
    /// Render headless and save to this file (format by extension, e.g. PNG).
    #[arg(short, long)]
    pub output: Option<PathBuf>,
    /// Fixed indices for axes beyond the 2D plane of an ND cube (comma-separated).
    #[arg(long, value_delimiter = ',')]
    pub plane: Vec<usize>,
    /// Colour map name (e.g. viridis, magma, inferno, plasma, turbo, cividis).
    #[arg(long, default_value = "viridis")]
    pub cmap: String,
    /// Value scaling.
    #[arg(long, value_enum, default_value_t = ScaleArg::Linear)]
    pub scale: ScaleArg,
    /// Linear-threshold half-width for `--scale symlog`.
    #[arg(long, default_value_t = 1.0)]
    pub linthresh: f32,
    /// Lower normalization bound (in scaled space).
    #[arg(long)]
    pub vmin: Option<f32>,
    /// Upper normalization bound (in scaled space).
    #[arg(long)]
    pub vmax: Option<f32>,
    /// Output width in pixels (default: the image's NAXIS1).
    #[arg(long)]
    pub width: Option<u32>,
    /// Output height in pixels (default: the image's NAXIS2).
    #[arg(long)]
    pub height: Option<u32>,
    /// Resampling filter (nearest, triangle, catmullrom, gaussian, lanczos3).
    #[arg(long, default_value = "nearest")]
    pub sampling: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::common::CommonArgs;
    use clap::{CommandFactory, Parser};

    /// Compose the parser exactly as the `lf-fits` binary does, to validate the
    /// argument definitions (no short/long conflicts, etc.).
    #[derive(Parser, Debug)]
    struct TestCli {
        #[command(flatten)]
        common: CommonArgs,
        #[command(flatten)]
        fits: FitsArgs,
    }

    #[test]
    fn clap_definition_is_valid() {
        TestCli::command().debug_assert();
    }

    #[test]
    fn parses_path_and_flags() {
        let cli = TestCli::try_parse_from(["lf-fits", "img.fits", "-H", "--cmap", "magma"]);
        let cli = cli.unwrap();
        assert_eq!(cli.fits.path, PathBuf::from("img.fits"));
        assert!(cli.fits.header);
        assert_eq!(cli.fits.cmap, "magma");
        assert_eq!(cli.fits.scale, ScaleArg::Linear);
    }
}
