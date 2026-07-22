#![allow(dead_code)]
#![allow(clippy::needless_return, clippy::result_large_err, clippy::needless_range_loop)]

pub mod diagram;
pub mod topology;
pub(crate) mod util;

use crate::diagram::DiagramContainer;
pub use crate::{
    diagram::{DiagramGenerator, DiagramSelector},
    util::InputError,
};
pub(crate) use model::{Model, ModelBase, ModelError};

/// Convenience function for the generation of Feynman diagrams.
///
/// See the documentation of the [`DiagramGenerator`] for details.
///
/// # Examples
/// ```rust
/// # use feyngraph_core::generate_diagrams;
/// # use model::Model;
/// let diags = generate_diagrams(&["u", "u~"], &["g"; 3], 2, Model::sm(), Default::default());
/// ```
pub fn generate_diagrams<M: ModelBase>(
    particles_in: &[&str],
    particles_out: &[&str],
    n_loops: usize,
    model: Model<M>,
    selector: DiagramSelector<M>,
) -> Result<DiagramContainer<M>, ModelError> {
    return Ok(DiagramGenerator::new(particles_in, particles_out, n_loops, model, Some(selector))?.generate());
}
