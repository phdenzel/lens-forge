// SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
// SPDX-FileNotice: Part of lens-forge
// SPDX-License-Identifier: GPL-3.0-or-later
//! The [`ExpansionOps`] scheme: two expansion carriers plus the six translation
//! operators a tree-driven FMM relays between them.
//!
//! An *outgoing* expansion sends a box's sources' influence outward; an
//! *incoming* expansion receives it at a target. The [`TreeFmm`](super::TreeFmm)
//! driver is written once against this trait, so a backend supplies only the
//! kernel-specific operators: the analytic [`Cauchy`](super::fmm::Cauchy) scheme
//! uses closed-form complex expansions, a future KIFMM equivalent densities.

use crate::fastsum::{Jet, Monopole};
use crate::grid::Point2D;
use crate::tree::TreeBox;

/// Kernel-specific expansion carriers and the operators that translate between
/// them; the [`TreeFmm`](super::TreeFmm) driver supplies the tree and traversal.
pub(crate) trait ExpansionOps {
    /// Outgoing (far-field) expansion — a box's sources seen from outside.
    type Outgoing: Clone;
    /// Incoming (local) expansion — the far field received at a box.
    type Incoming: Clone;
    /// Precomputed, kernel- and scale-dependent operator data for one build.
    type Ctx;

    /// Build the shared operator context for a tree of the given geometry.
    ///
    /// # Arguments
    /// * `root_half` - half side-length of the root box.
    /// * `max_depth` - deepest tree level (the scale range to precompute).
    /// * `order` - expansion order / proxy points per side.
    /// * `eps2` - squared softening length.
    fn context(root_half: f64, max_depth: usize, order: usize, eps2: f64) -> Self::Ctx;

    /// S2M: leaf `sources` → the outgoing expansion of box `b`.
    ///
    /// # Arguments
    /// * `ctx` - operator context.
    /// * `b` - the leaf box.
    /// * `sources` - the box's monopoles.
    fn s2m(ctx: &Self::Ctx, b: TreeBox, sources: &[Monopole]) -> Self::Outgoing;

    /// M2M: shift `child`'s outgoing expansion up to parent box `b`, merging the
    /// result into `acc` (co-centred accumulation over the parent's children).
    ///
    /// # Arguments
    /// * `ctx` - operator context.
    /// * `b` - the parent box.
    /// * `child` - a child's outgoing expansion.
    /// * `acc` - running parent expansion, updated in place.
    fn m2m(ctx: &Self::Ctx, b: TreeBox, child: &Self::Outgoing, acc: &mut Option<Self::Outgoing>);

    /// An empty incoming expansion centred on box `b`.
    ///
    /// # Arguments
    /// * `ctx` - operator context.
    /// * `b` - the box.
    fn zero_incoming(ctx: &Self::Ctx, b: TreeBox) -> Self::Incoming;

    /// L2L: shift `parent`'s incoming expansion down to child box `b`,
    /// accumulating into `into`.
    ///
    /// # Arguments
    /// * `ctx` - operator context.
    /// * `b` - the child box.
    /// * `parent` - the parent's incoming expansion.
    /// * `into` - the child's incoming expansion, updated in place.
    fn l2l(ctx: &Self::Ctx, b: TreeBox, parent: &Self::Incoming, into: &mut Self::Incoming);

    /// M2L: convert a well-separated source box `src`'s outgoing expansion into
    /// box `b`'s incoming expansion (V-list), accumulating into `into`.
    ///
    /// # Arguments
    /// * `ctx` - operator context.
    /// * `b` - the target box.
    /// * `src` - the well-separated source box.
    /// * `out` - the source box's outgoing expansion.
    /// * `into` - box `b`'s incoming expansion, updated in place.
    fn m2l(
        ctx: &Self::Ctx,
        b: TreeBox,
        src: TreeBox,
        out: &Self::Outgoing,
        into: &mut Self::Incoming,
    );

    /// P2L: add coarse-leaf `sources` (X-list) directly to box `b`'s incoming
    /// expansion, accumulating into `into`.
    ///
    /// # Arguments
    /// * `ctx` - operator context.
    /// * `b` - the target box.
    /// * `sources` - the coarse-leaf monopoles.
    /// * `into` - box `b`'s incoming expansion, updated in place.
    fn p2l(ctx: &Self::Ctx, b: TreeBox, sources: &[Monopole], into: &mut Self::Incoming);

    /// L2P: evaluate the incoming expansion's jet at `theta`.
    ///
    /// # Arguments
    /// * `ctx` - operator context.
    /// * `inc` - the box's incoming expansion.
    /// * `theta` - query point near the expansion centre.
    fn l2p(ctx: &Self::Ctx, inc: &Self::Incoming, theta: Point2D) -> Jet;

    /// M2P: evaluate a far-field outgoing expansion's jet at `theta` (used for
    /// the root, i.e. queries outside the tree's root box).
    ///
    /// # Arguments
    /// * `ctx` - operator context.
    /// * `out` - a far-field outgoing expansion.
    /// * `theta` - query point outside the expansion's field.
    fn m2p(ctx: &Self::Ctx, out: &Self::Outgoing, theta: Point2D) -> Jet;
}
