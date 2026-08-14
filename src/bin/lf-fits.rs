// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! The `lf-fits` binary: inspect or plot a FITS image (`lf fits ...`).
//!
//! Reads a FITS file with [`Fits::open`], then either prints a metadata summary
//! (the default) or the raw header cards, and/or renders the stored image —
//! either to an interactive window ([`view_grid`]) or, with `--output`, headless
//! to a PNG via the [`viz`](lens_forge::viz) raster path. Argument definitions
//! live in [`lens_forge::cli::fits`]; the run logic lives here.
// Lets the interactive-window dispatcher opt out of coverage (`#[coverage(off)]`).
#![feature(coverage_attribute)]

use clap::Parser;
use lens_forge::cli::common::CommonArgs;
use lens_forge::cli::fits::{FitsArgs, ScaleArg};
use lens_forge::grid::PixelGrid;
use lens_forge::io::fits::{Fits, HduKind, LinearWcs};
use lens_forge::viz::render::context::RasterContext;
use lens_forge::viz::render::{RenderOptions, rasterize};
use lens_forge::viz::scale::RenderScale;
use lens_forge::viz::view::view_grid;
use lens_forge::{LensForgeError, Result};
use std::num::NonZeroU32;

/// Inspect or plot a FITS image.
#[derive(Parser, Debug)]
#[command(name = "lf-fits", version, about = "Inspect or plot a FITS image")]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,
    #[command(flatten)]
    fits: FitsArgs,
}

fn main() {
    let cli = Cli::parse();
    if let Err(e) = run(&cli.fits, &cli.common) {
        eprintln!("lf-fits: {e}");
        std::process::exit(1);
    }
}

/// Run the `fits` subcommand.
///
/// # Arguments
/// * `args` - the parsed subcommand arguments.
/// * `common` - the shared flags (verbosity).
fn run(args: &FitsArgs, common: &CommonArgs) -> Result<()> {
    let fits = Fits::open(&args.path)?;

    let want_plot = args.plot || args.output.is_some();
    let want_header = args.header;
    // With no action requested, default to the metadata summary.
    let want_info = args.info || (!want_header && !want_plot);

    if want_info {
        print!("{}", format_info(&fits));
    }
    if want_header {
        println!("{}", format_header(&fits, args.hdu)?);
    }
    if want_plot {
        plot(&fits, args, common)?;
    }
    Ok(())
}

/// The [`RenderScale`] selected by `--scale` (and `--linthresh`).
fn render_scale(args: &FitsArgs) -> RenderScale {
    match args.scale {
        ScaleArg::Linear => RenderScale::Linear,
        ScaleArg::Log10 => RenderScale::Log10,
        ScaleArg::Symlog => RenderScale::SymLog {
            linthresh: args.linthresh,
        },
    }
}

/// Build the pixel grid and render options for the selected HDU/plane.
fn prepare(fits: &Fits, args: &FitsArgs) -> Result<(PixelGrid, RenderOptions)> {
    let image = fits.image(args.hdu).ok_or_else(|| {
        LensForgeError::invalid_parameter(format!("HDU {} has no image to plot", args.hdu))
    })?;
    let nx = image.naxes.first().copied().unwrap_or(1) as u32;
    let ny = image.naxes.get(1).copied().unwrap_or(1) as u32;

    let opt = RenderOptions {
        width: NonZeroU32::new(args.width.unwrap_or(nx).max(1)).unwrap(),
        height: NonZeroU32::new(args.height.unwrap_or(ny).max(1)).unwrap(),
        scale: render_scale(args),
        vmin: args.vmin,
        vmax: args.vmax,
        gradient_name: args.cmap.clone(),
        sampling_type: args.sampling.clone(),
        ..Default::default()
    };
    opt.validate()?;

    let pg = fits.to_pixelgrid(args.hdu, &args.plane)?;
    Ok((pg, opt))
}

/// Render the grid headless and save it to `out`.
fn save(pg: &PixelGrid, opt: &RenderOptions, out: &std::path::Path, common: &CommonArgs) -> Result<()> {
    let raster = rasterize(pg)?;
    let img = RasterContext::new().render(&raster, opt)?;
    img.save(out)
        .map_err(|e| LensForgeError::io_error(format!("failed to save {out:?}: {e}")))?;
    if common.is_loud() {
        println!(
            "Saved {}x{} image to {}",
            opt.width_u32(),
            opt.height_u32(),
            out.display()
        );
    }
    Ok(())
}

/// Render the selected HDU's image, to a file (`--output`) or a window.
///
/// The interactive-window branch needs a GPU/display, so it cannot run in a
/// headless test; the substantive logic lives in [`prepare`]/[`save`], which
/// are covered, and this thin dispatcher is excluded from coverage.
#[coverage(off)]
fn plot(fits: &Fits, args: &FitsArgs, common: &CommonArgs) -> Result<()> {
    let (pg, opt) = prepare(fits, args)?;
    match &args.output {
        Some(out) => save(&pg, &opt, out, common),
        None => {
            view_grid(&pg, &opt);
            Ok(())
        }
    }
}

/// A human-readable whole-file metadata summary.
///
/// # Arguments
/// * `fits` - the parsed file.
fn format_info(fits: &Fits) -> String {
    let mut s = format!("FITS file: {} HDU(s)\n", fits.len());
    for (i, hdu) in fits.hdus.iter().enumerate() {
        let kind = match &hdu.kind {
            HduKind::Primary => "Primary".to_string(),
            HduKind::Image => "Image".to_string(),
            HduKind::Other(name) => format!("Other({name})"),
        };
        s.push_str(&format!("\nHDU {i}: {kind}\n"));
        match &hdu.image {
            Some(img) => {
                let dims = img
                    .naxes
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(" x ");
                s.push_str(&format!("  BITPIX = {}\n", img.bitpix));
                s.push_str(&format!("  NAXIS  = {} [{}]\n", img.ndim(), dims));
                if img.bscale != 1.0 || img.bzero != 0.0 {
                    s.push_str(&format!("  BSCALE = {}, BZERO = {}\n", img.bscale, img.bzero));
                }
                if let Some(b) = img.blank {
                    s.push_str(&format!("  BLANK  = {b}\n"));
                }
                let wcs = if LinearWcs::from_header(&hdu.header).is_some() {
                    "present (linear)"
                } else {
                    "none"
                };
                s.push_str(&format!("  WCS    = {wcs}\n"));
                if let Some((mn, mx, mean)) = finite_stats(img.as_f64_array().iter().copied()) {
                    s.push_str(&format!("  min/max/mean = {mn:.6} / {mx:.6} / {mean:.6}\n"));
                }
            }
            None => {
                s.push_str(&format!(
                    "  (no image data; BITPIX = {}, NAXIS = {})\n",
                    hdu.header.get_int("BITPIX").unwrap_or(0),
                    hdu.header.get_int("NAXIS").unwrap_or(0),
                ));
            }
        }
    }
    s
}

/// The selected HDU's header rendered as its 80-column card images.
///
/// # Arguments
/// * `fits` - the parsed file.
/// * `hdu` - the HDU index.
fn format_header(fits: &Fits, hdu: usize) -> Result<String> {
    let h = fits
        .hdus
        .get(hdu)
        .ok_or_else(|| LensForgeError::invalid_parameter(format!("no HDU at index {hdu}")))?;
    let mut s = String::new();
    for card in h.header.cards() {
        let image = card.to_card_image();
        s.push_str(String::from_utf8_lossy(&image).trim_end());
        s.push('\n');
    }
    s.push_str("END");
    Ok(s)
}

/// `(min, max, mean)` over the finite values, or `None` if there are none.
fn finite_stats(vals: impl Iterator<Item = f64>) -> Option<(f64, f64, f64)> {
    let (mut mn, mut mx, mut sum, mut n) = (f64::INFINITY, f64::NEG_INFINITY, 0.0, 0usize);
    for v in vals {
        if v.is_finite() {
            mn = mn.min(v);
            mx = mx.max(v);
            sum += v;
            n += 1;
        }
    }
    (n > 0).then(|| (mn, mx, sum / n as f64))
}

#[cfg(test)]
mod tests {
    use super::*;
    use lens_forge::grid::{BBox, Grid2D, PixelGrid};
    use lens_forge::io::fits::{Card, FitsData, FitsImage, Hdu, Header};
    use std::path::PathBuf;

    fn tmp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("lf_fits_{}_{name}", std::process::id()))
    }

    fn write_sample(path: &PathBuf) {
        let grid = Grid2D::new(4, 3, BBox::from_lims(-2.0, 2.0, -1.5, 1.5)).unwrap();
        let pg = PixelGrid::from_func(grid, |x, y| x + 10.0 * y);
        Fits::from_pixelgrid(&pg, Some(-64))
            .unwrap()
            .write(path)
            .unwrap();
    }

    fn base_args(path: PathBuf) -> FitsArgs {
        FitsArgs {
            path,
            hdu: 0,
            header: false,
            info: false,
            plot: false,
            output: None,
            plane: vec![],
            cmap: "viridis".into(),
            scale: ScaleArg::Linear,
            linthresh: 1.0,
            vmin: None,
            vmax: None,
            width: None,
            height: None,
            sampling: "nearest".into(),
        }
    }

    #[test]
    fn info_and_header_formatting() {
        let path = tmp_path("info.fits");
        write_sample(&path);
        let fits = Fits::open(&path).unwrap();

        let info = format_info(&fits);
        assert!(info.contains("HDU 0: Primary"));
        assert!(info.contains("BITPIX = -64"));
        assert!(info.contains("4 x 3")); // NAXIS1 x NAXIS2 = nx x ny
        assert!(info.contains("WCS    = present (linear)"));
        assert!(info.contains("min/max/mean"));

        let hdr = format_header(&fits, 0).unwrap();
        assert!(hdr.starts_with("SIMPLE"));
        assert!(hdr.contains("NAXIS1"));
        assert!(hdr.trim_end().ends_with("END"));

        assert!(format_header(&fits, 9).is_err()); // out-of-range HDU
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn output_saves_png_at_requested_size() {
        let path = tmp_path("plot.fits");
        write_sample(&path);
        let out = tmp_path("plot.png");

        let mut args = base_args(path.clone());
        args.output = Some(out.clone());
        args.width = Some(16);
        args.height = Some(12);
        run(&args, &CommonArgs::default()).unwrap();

        let img = image::open(&out).unwrap();
        assert_eq!(img.width(), 16);
        assert_eq!(img.height(), 12);

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&out);
    }

    #[test]
    fn default_action_is_info() {
        let path = tmp_path("default.fits");
        write_sample(&path);
        // No action flags: should succeed (prints the info summary).
        run(&base_args(path.clone()), &CommonArgs::default()).unwrap();
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn run_with_header_ok() {
        let path = tmp_path("runhdr.fits");
        write_sample(&path);
        let mut args = base_args(path.clone());
        args.header = true;
        run(&args, &CommonArgs::default()).unwrap();
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn render_scale_maps_every_variant() {
        let mut a = base_args(PathBuf::from("x"));
        a.scale = ScaleArg::Linear;
        assert!(matches!(render_scale(&a), RenderScale::Linear));
        a.scale = ScaleArg::Log10;
        assert!(matches!(render_scale(&a), RenderScale::Log10));
        a.scale = ScaleArg::Symlog;
        a.linthresh = 2.5;
        assert!(matches!(
            render_scale(&a),
            RenderScale::SymLog { linthresh } if (linthresh - 2.5).abs() < 1e-6
        ));
    }

    #[test]
    fn prepare_without_image_errors() {
        // A dataless primary HDU cannot be rendered.
        let fits = Fits {
            hdus: vec![Hdu::from_parts(Header::new(), None, HduKind::Primary)],
        };
        assert!(prepare(&fits, &base_args(PathBuf::from("unused"))).is_err());
    }

    #[test]
    fn format_info_covers_scaling_and_dataless() {
        // Image carrying non-default scaling and no WCS.
        let img =
            FitsImage::new(vec![2, 2], FitsData::I16(vec![0, 1, 2, 3]), 2.0, 100.0, Some(-1))
                .unwrap();
        let s = format_info(&Fits::from_image(img));
        assert!(s.contains("BSCALE = 2"));
        assert!(s.contains("BZERO = 100"));
        assert!(s.contains("BLANK  = -1"));
        assert!(s.contains("WCS    = none"));

        // A dataless HDU takes the no-image branch.
        let mut header = Header::new();
        header.push(Card::integer("BITPIX", 8));
        header.push(Card::integer("NAXIS", 0));
        let fits = Fits {
            hdus: vec![Hdu::from_parts(header, None, HduKind::Primary)],
        };
        assert!(format_info(&fits).contains("no image data"));
    }

    #[test]
    fn format_info_labels_image_and_other_hdus() {
        let img = FitsImage::from_field_array(Some(-64), &ndarray::array![[1.0, 2.0]]).unwrap();
        let fits = Fits {
            hdus: vec![
                Hdu::primary(img.clone()),
                Hdu::image_ext(img),
                Hdu::from_parts(Header::new(), None, HduKind::Other("BINTABLE".into())),
            ],
        };
        let s = format_info(&fits);
        assert!(s.contains("HDU 1: Image"));
        assert!(s.contains("HDU 2: Other(BINTABLE)"));
    }
}
