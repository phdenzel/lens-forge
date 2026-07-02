#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
# SPDX-FileNotice: Part of lens-forge
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Build the API docs with LaTeX math rendering (KaTeX). Mirrors the docs.rs
# configuration in Cargo.toml's [package.metadata.docs.rs].
#
# Usage:
#   ./docs/build-docs.sh          # build docs
#   ./docs/build-docs.sh --open   # build and open in a browser
#
# Extra arguments are forwarded to `cargo doc`.
set -euo pipefail

# Resolve paths relative to this script, then run from the crate root (where
# Cargo.toml lives) so `cargo doc` works regardless of the caller's cwd.
script_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$script_dir/.."

# Inject the KaTeX header into every rustdoc invocation for this build.
export RUSTDOCFLAGS="--html-in-header $script_dir/katex-header.html"

# Prefer the Nix dev shell (provides libiconv/wgpu system deps on macOS);
# fall back to a bare cargo if Nix is unavailable.
if command -v nix >/dev/null 2>&1; then
    exec nix develop --command cargo doc --no-deps "$@"
else
    exec cargo doc --no-deps "$@"
fi
