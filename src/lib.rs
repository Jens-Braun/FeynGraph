#![allow(dead_code)]
#![allow(clippy::needless_return, clippy::result_large_err, clippy::needless_range_loop)]

mod bindings;
pub mod diagram;
mod drawing;
pub mod model;
pub mod topology;
pub(crate) mod util;

pub mod prelude {
    pub use crate::{
        diagram::{DiagramGenerator, filter::DiagramSelector},
        model::Model,
    };
}
