//! This crate contains the native Rust interface of FeynGraph, a modern Feynman diagram generator.

#![allow(dead_code)]
#![allow(clippy::needless_return, clippy::result_large_err, clippy::needless_range_loop)]

mod bindings;
pub mod diagram;
mod drawing;
pub mod model;
pub mod topology;
pub(crate) mod util;

use crate::{diagram::DiagramContainer, model::ModelError};
pub use crate::{
    diagram::{DiagramGenerator, DiagramSelector},
    model::Model,
};

/// Convenience function for the generation of Feynman diagrams.
///
/// See the documentation of the [`DiagramGenerator`] for details.
///
/// # Examples
/// ```rust
/// use feyngraph::generate_diagrams;
/// let diags = generate_diagrams(&["u", "u~"], &["g"; 3], 2, Default::default(), Default::default());
/// ```
pub fn generate_diagrams(
    particles_in: &[&str],
    particles_out: &[&str],
    n_loops: usize,
    model: Model,
    selector: DiagramSelector,
) -> Result<DiagramContainer, ModelError> {
    return Ok(DiagramGenerator::new(particles_in, particles_out, n_loops, model, Some(selector))?.generate());
}
