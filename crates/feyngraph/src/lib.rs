//! This crate contains the native Rust interface of FeynGraph, a modern Feynman diagram generation toolkit.

pub mod prelude {
    //! This module allows easy import of the most used types and traits.
    #[cfg(feature = "drawing")]
    pub use crate::drawing::{Draw, DrawGrid};
    pub use crate::{
        diagram::{DiagramGenerator, DiagramSelector, generate_diagrams},
        model::Model,
    };
}

pub mod diagram {
    //! Central module for the generation of Feynman diagrams.
    //!
    //! This module contains the [`DiagramGenerator`], which handles the generation of Feynman diagrams given a
    //! [`Model`](crate::model::Model) and optionally a [`DiagramSelector`] restricting which diagrams are generated.
    pub use feyngraph_core::diagram::{
        DiagramContainer, DiagramGenerator, DiagramSelector,
        view::{DiagramView, LegView, PropagatorView, VertexView},
    };
    pub use feyngraph_core::generate_diagrams;
}

pub mod model {
    //! A physical model used for diagram generation and drawing.
    pub use feyngraph_model::{
        LineStyle, Model, ModelBase, ModelError, ParticleBase, ParticleColor, ParticleDraw, VertexBase,
    };

    pub mod ufo {
        pub use feyngraph_model::{UFOModel, UFOParticle, UFOVertex};
    }
}

pub mod topology {
    //! Central module for the generation of unassigned graphs, called _topologies_.
    //!
    //! The central object of this module is the [`TopologyGenerator`], which handles the generation
    //! of the topologies given a [`TopologyModel`] and optionally a [`TopologySelector`]
    //! restricting which topologies are generated.
    pub use feyngraph_core::topology::{Edge, Node, Topology, TopologyContainer, TopologyGenerator, TopologySelector};
    pub use feyngraph_model::TopologyModel;
}

#[cfg(feature = "drawing")]
pub mod drawing {
    pub use feyngraph_drawing::{
        Anchor, Backend, Color, Decoration, DecorationKind, Draw, DrawGrid, PathStyle, SVGBackend, Stroke, Theme,
        TikzBackend, TypstBackend,
    };
}
