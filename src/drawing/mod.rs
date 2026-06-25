use either::Either;
use itertools::Itertools;
use layout::Layout;
use std::ops::{Bound, RangeBounds};
use std::path::Path;

use crate::{
    diagram::{DiagramContainer, view::DiagramView},
    model::LineStyle,
    topology::{Topology, TopologyContainer},
};
use backend::Backend;
use canvas::{Canvas, CanvasGrid};
use components::Segment;
use layout::{DiagramLayout, TopologyLayout};

mod backend;
mod canvas;
mod components;
mod consts;
mod layout;
mod math;
mod style;
mod util;

pub(crate) use backend::SVGBackend;
pub(crate) use backend::TikzBackend;
pub(crate) use backend::TypstBackend;
pub(crate) use style::{Anchor, Color, Decoration, DecorationKind, PathStyle, Stroke, Theme};

impl Topology {
    fn draw<'a, B: Backend>(&self, mut canvas: Canvas<'a, B>) {
        let layout = TopologyLayout::from(self).layout();
        for i in 0..self.n_external {
            let x = &layout[i];
            canvas.push_label(
                i.to_string(),
                *x,
                if i < self.n_external / 2 {
                    Anchor::Right
                } else {
                    Anchor::Left
                },
                true,
            );
        }
        let (v_min, v_max) = layout.bounding_box();
        let scale = (v_max - v_min).norm();
        for (nodes, chunk) in &self.edges.iter().chunk_by(|e| e.connected_nodes) {
            if nodes[0] == nodes[1] {
                let x = &layout[nodes[0]];
                let neighbors = self.nodes[nodes[0]]
                    .adjacent_nodes
                    .iter()
                    .filter(|n| **n != nodes[0])
                    .map(|n| layout[*n])
                    .collect_vec();
                for path in util::self_paths(*x, &neighbors, scale, chunk.count()).into_iter() {
                    canvas.push_path(*x, path, PathStyle::default(), LineStyle::None, None);
                }
            } else {
                let x1 = &layout[nodes[0]];
                let x2 = &layout[nodes[1]];
                for path in util::multi_path(*x1, *x2, chunk.count()).into_iter() {
                    canvas.push_path(*x1, path, PathStyle::default(), LineStyle::None, None);
                }
            }
        }
        canvas.finish();
    }

    /// Draw the topology in the SVG format and write the result to the file `path`.
    pub fn draw_svg(&self, path: impl AsRef<Path>) -> Result<(), std::io::Error> {
        let mut grid = CanvasGrid::<SVGBackend>::new(1, 1);
        self.draw(grid.canvas(0, 0));
        std::fs::write(path, grid.finish())?;
        Ok(())
    }

    /// Draw the topology in SVG format and return the result as string.
    pub fn draw_svg_string(&self) -> String {
        let mut grid = CanvasGrid::<SVGBackend>::new(1, 1);
        self.draw(grid.canvas(0, 0));
        grid.finish()
    }

    /// Draw the topology in the SVG format and write the result to the file `path`.
    pub fn draw_tikz(&self, path: impl AsRef<Path>) -> Result<(), std::io::Error> {
        let mut grid = CanvasGrid::<TikzBackend>::new(1, 1);
        self.draw(grid.canvas(0, 0));
        std::fs::write(path, grid.finish())?;
        Ok(())
    }

    /// Draw the topology in SVG format and return the result as string.
    pub fn draw_tikz_string(&self) -> String {
        let mut grid = CanvasGrid::<TikzBackend>::new(1, 1);
        self.draw(grid.canvas(0, 0));
        grid.finish()
    }

    /// Draw the topology in the SVG format and write the result to the file `path`.
    pub fn draw_typst(&self, path: impl AsRef<Path>) -> Result<(), std::io::Error> {
        let mut grid = CanvasGrid::<TypstBackend>::new(1, 1);
        self.draw(grid.canvas(0, 0));
        std::fs::write(path, grid.finish())?;
        Ok(())
    }

    /// Draw the topology in SVG format and return the result as string.
    pub fn draw_typst_string(&self) -> String {
        let mut grid = CanvasGrid::<TypstBackend>::new(1, 1);
        self.draw(grid.canvas(0, 0));
        grid.finish()
    }
}

impl TopologyContainer {
    /// Draw the topologies with indices `topologies` in SVG format in a grid on a single canvas. If specified, the
    /// grid will have `n_cols` topologies per row, otherwise four.
    pub fn draw<B: Backend>(&self, topologies: &[usize], n_cols: Option<usize>) -> B::Output {
        let n_topos = topologies.len();
        let n_cols = if let Some(n_cols) = n_cols {
            n_cols
        } else if n_topos < 4 {
            n_topos
        } else {
            4
        };
        let n_rows = n_topos.div_ceil(n_cols);
        let mut grid = CanvasGrid::<B>::new(n_cols, n_rows);
        for (i, topo_id) in topologies.iter().enumerate() {
            grid.draw_title(i / n_cols, i % n_cols, &format!("T{}", topo_id));
            self.data[*topo_id].draw(grid.canvas(i / n_cols, i % n_cols));
        }
        return grid.finish();
    }

    /// Draw the topologies with indices in `range` in SVG format on a single canvas. If specified, the
    /// grid will have `n_cols` topologies per row, otherwise four.
    pub fn draw_range<B: Backend>(&self, range: impl RangeBounds<usize>, n_cols: Option<usize>) -> B::Output {
        let min = match range.start_bound() {
            Bound::Included(n) => *n,
            Bound::Excluded(n) if *n > 0 => *n - 1,
            _ => 0,
        };
        let max = match range.end_bound() {
            Bound::Included(n) => *n,
            Bound::Excluded(n) => {
                if *n > 0 {
                    *n - 1
                } else {
                    0
                }
            }
            Bound::Unbounded => self.data.len(),
        };
        let n_topos = max - min;
        let n_cols = if let Some(n_cols) = n_cols {
            n_cols
        } else if n_topos < 4 {
            n_topos
        } else {
            4
        };
        let n_rows = n_topos.div_ceil(n_cols);
        let mut grid = CanvasGrid::<B>::new(n_cols, n_rows);
        for (i, topo_id) in (min..=max).enumerate() {
            grid.draw_title(i / n_cols, i % n_cols, &format!("T{}", topo_id));
            self.data[topo_id].draw(grid.canvas(i / n_cols, i % n_cols));
        }
        return grid.finish();
    }

    /// Draw the topologies with indices `topologies` in the SVG format in a grid on a single canvas. If specified, the
    /// grid will have `n_cols` topologies per row, otherwise four.
    /// The result is returned as a single string.
    pub fn draw_svg_string(&self, topologies: &[usize], n_cols: Option<usize>) -> String {
        self.draw::<SVGBackend>(topologies, n_cols)
    }

    /// Draw the topologies with indices `topologies` in the SVG format in a grid on a single canvas. If specified, the
    /// grid will have `n_cols` topologies per row, otherwise four.
    /// The result is written to the file `path`.
    pub fn draw_svg(&self, path: impl AsRef<Path>) -> Result<(), std::io::Error> {
        std::fs::write(path, self.draw_svg_range_string(.., None))?;
        Ok(())
    }

    /// Draw the topologies with indices `topologies` in the Typst format in a grid on a single canvas. If specified, the
    /// grid will have `n_cols` topologies per row, otherwise four.
    /// The result is returned as a single string.
    pub fn draw_typst_string(&self, topologies: &[usize], n_cols: Option<usize>) -> String {
        self.draw::<TypstBackend>(topologies, n_cols)
    }

    /// Draw the topologies with indices `topologies` in the Typst format in a grid on a single canvas. If specified, the
    /// grid will have `n_cols` topologies per row, otherwise four.
    /// The result is written to the file `path`.
    pub fn draw_typst(&self, path: impl AsRef<Path>) -> Result<(), std::io::Error> {
        std::fs::write(path, self.draw_typst_range_string(.., None))?;
        Ok(())
    }

    /// Draw the topologies with indices `topologies` in the TikZ format in a grid on a single canvas. If specified, the
    /// grid will have `n_cols` topologies per row, otherwise four.
    /// The result is returned as a single string.
    pub fn draw_tikz_string(&self, topologies: &[usize], n_cols: Option<usize>) -> String {
        self.draw::<TikzBackend>(topologies, n_cols)
    }

    /// Draw the topologies with indices `topologies` in the TikZ format in a grid on a single canvas. If specified, the
    /// grid will have `n_cols` topologies per row, otherwise four.
    /// The result is written to the file `path`.
    pub fn draw_tikz(&self, path: impl AsRef<Path>) -> Result<(), std::io::Error> {
        std::fs::write(path, self.draw_tikz_range_string(.., None))?;
        Ok(())
    }

    /// Draw the topologies with indices in `range` in the given format on a single canvas. If specified, the
    /// grid will have `n_cols` topologies per row, otherwise four.
    pub fn draw_svg_range_string(&self, range: impl RangeBounds<usize>, n_cols: Option<usize>) -> String {
        self.draw_range::<SVGBackend>(range, n_cols)
    }

    /// Draw the topologies with indices in `range` in the given format on a single canvas. If specified, the
    /// grid will have `n_cols` topologies per row, otherwise four.
    pub fn draw_typst_range_string(&self, range: impl RangeBounds<usize>, n_cols: Option<usize>) -> String {
        self.draw_range::<TypstBackend>(range, n_cols)
    }

    /// Draw the topologies with indices in `range` in the given format on a single canvas. If specified, the
    /// grid will have `n_cols` topologies per row, otherwise four.
    pub fn draw_tikz_range_string(&self, range: impl RangeBounds<usize>, n_cols: Option<usize>) -> String {
        self.draw_range::<TikzBackend>(range, n_cols)
    }
}

impl DiagramView<'_> {
    pub(crate) fn draw<'a, B: Backend>(&self, mut canvas: Canvas<'a, B>) {
        let theme = Theme::get_global();
        let layout = DiagramLayout::from(self).layout();
        let n_ext = self.n_ext();
        if self.diagram.vertices.is_empty() {
            let l = self.incoming().next().unwrap();
            let x1 = &layout[0];
            let x2 = &layout[1];
            canvas.push_label(
                format!("{}({})", B::particle_name(l.particle()), 0),
                *x1,
                Anchor::Right,
                true,
            );
            canvas.push_label(
                format!("{}({})", B::particle_name(l.particle()), 1),
                *x1,
                Anchor::Right,
                true,
            );
            if l.particle().is_anti() {
                canvas.push_path(
                    *x2,
                    Segment::Line(*x1),
                    theme.get_particle_or_default(l.particle().name()),
                    l.particle().linestyle,
                    None,
                );
            } else {
                canvas.push_path(
                    *x1,
                    Segment::Line(*x2),
                    theme.get_particle_or_default(l.particle().name()),
                    l.particle().linestyle,
                    None,
                );
            }
        } else {
            for l in self.incoming() {
                let x1 = &layout[l.leg_index];
                let x2 = &layout[l.leg.vertex + n_ext];
                canvas.push_label(
                    format!("{}({})", B::particle_name(l.particle()), l.leg_index),
                    *x1,
                    Anchor::Right,
                    true,
                );
                if l.particle().is_anti() {
                    canvas.push_path(
                        *x2,
                        Segment::Line(*x1),
                        theme.get_particle_or_default(l.particle().name()),
                        l.particle().linestyle,
                        None,
                    );
                } else {
                    canvas.push_path(
                        *x1,
                        Segment::Line(*x2),
                        theme.get_particle_or_default(l.particle().name()),
                        l.particle().linestyle,
                        None,
                    );
                }
            }
            for l in self.outgoing() {
                let x1 = &layout[l.leg_index];
                let x2 = &layout[l.leg.vertex + n_ext];
                canvas.push_label(
                    format!("{}({})", B::particle_name(l.particle()), l.leg_index),
                    *x1,
                    Anchor::Left,
                    true,
                );
                if !l.particle().is_anti() {
                    canvas.push_path(
                        *x2,
                        Segment::Line(*x1),
                        theme.get_particle_or_default(l.particle().name()),
                        l.particle().linestyle,
                        None,
                    );
                } else {
                    canvas.push_path(
                        *x1,
                        Segment::Line(*x2),
                        theme.get_particle_or_default(l.particle().name()),
                        l.particle().linestyle,
                        None,
                    );
                }
            }
        }
        let (v_min, v_max) = layout.bounding_box();
        let scale = (v_max - v_min).norm();
        for (vertices, chunk) in &self.propagators().chunk_by(|p| p.propagator.vertices) {
            let particles = chunk.collect::<Vec<_>>();
            if vertices[0] == vertices[1] {
                let x = layout[vertices[0] + n_ext];
                let neighbors = self
                    .vertex(vertices[0])
                    .propagators()
                    .filter_map(|p| match p {
                        Either::Left(l) => Some(layout[l.leg_index]),
                        Either::Right(p) => {
                            if p.propagator.vertices[0] == vertices[0] && p.propagator.vertices[1] != vertices[0] {
                                Some(layout[p.propagator.vertices[1] + n_ext])
                            } else if p.propagator.vertices[1] == vertices[0] && p.propagator.vertices[0] != vertices[0]
                            {
                                Some(layout[p.propagator.vertices[0] + n_ext])
                            } else {
                                None
                            }
                        }
                    })
                    .collect::<Vec<_>>();
                for (mut path, p) in util::self_paths(x, &neighbors, scale, particles.len())
                    .into_iter()
                    .zip(particles.into_iter())
                {
                    if p.particle().is_anti() {
                        path.invert(x);
                    }
                    canvas.push_path(
                        x,
                        path,
                        theme.get_particle_or_default(p.particle().name()),
                        p.particle().linestyle,
                        Some(B::particle_name(p.particle()).to_owned()),
                    );
                }
            } else {
                let x1 = layout[vertices[0] + n_ext];
                let x2 = layout[vertices[1] + n_ext];
                for (mut path, p) in util::multi_path(x1, x2, particles.len())
                    .into_iter()
                    .zip(particles.into_iter())
                {
                    if p.particle().is_anti() {
                        path.invert(x1);
                        canvas.push_path(
                            x2,
                            path,
                            theme.get_particle_or_default(p.particle().name()),
                            p.particle().linestyle,
                            Some(B::particle_name(p.particle()).to_owned()),
                        );
                    } else {
                        canvas.push_path(
                            x1,
                            path,
                            theme.get_particle_or_default(p.particle().name()),
                            p.particle().linestyle,
                            Some(B::particle_name(p.particle()).to_owned()),
                        );
                    }
                }
            }
        }
        for v in self.vertices() {
            if let Some(d) = theme.get_vertex(v.interaction().name()) {
                canvas.push_decoration(layout[v.id() + n_ext], d.clone());
            }
        }
        canvas.finish();
    }

    /// Draw the diagram in the SVG format and return the result as string.
    pub fn draw_svg_string(&self) -> String {
        let mut grid = CanvasGrid::<SVGBackend>::new(1, 1);
        self.draw(grid.canvas(0, 0));
        grid.finish()
    }

    /// Draw the diagram in the SVG format and write the result to the file `path`.
    pub fn draw_svg(&self, path: impl AsRef<Path>) -> Result<(), std::io::Error> {
        let mut grid = CanvasGrid::<SVGBackend>::new(1, 1);
        self.draw(grid.canvas(0, 0));
        std::fs::write(path, grid.finish())?;
        Ok(())
    }

    /// Draw the diagram in Typst format and return the result as string.
    pub fn draw_typst_string(&self) -> String {
        let mut grid = CanvasGrid::<TypstBackend>::new(1, 1);
        self.draw(grid.canvas(0, 0));
        grid.finish()
    }

    /// Draw the diagram in Typst format and write the result to the file `path`.
    pub fn draw_typst(&self, path: impl AsRef<Path>) -> Result<(), std::io::Error> {
        let mut grid = CanvasGrid::<TypstBackend>::new(1, 1);
        self.draw(grid.canvas(0, 0));
        std::fs::write(path, grid.finish())?;
        Ok(())
    }

    /// Draw the diagram in TikZ format and return the result as string.
    pub fn draw_tikz_string(&self) -> String {
        let mut grid = CanvasGrid::<TikzBackend>::new(1, 1);
        self.draw(grid.canvas(0, 0));
        grid.finish()
    }

    /// Draw the diagram in TikZ format and write the result to the file `path`.
    pub fn draw_tikz(&self, path: impl AsRef<Path>) -> Result<(), std::io::Error> {
        let mut grid = CanvasGrid::<TikzBackend>::new(1, 1);
        self.draw(grid.canvas(0, 0));
        std::fs::write(path, grid.finish())?;
        Ok(())
    }
}

impl DiagramContainer {
    /// Draw the diagrams with indices `diagrams` in the SVG format in a grid on a single canvas. If specified, the
    /// grid will have `n_cols` diagrams per row, otherwise four.
    /// The result is returned as a single string.
    pub fn draw_svg_string(&self, diagrams: &[usize], n_cols: Option<usize>) -> String {
        self.draw::<SVGBackend>(diagrams, n_cols)
    }

    /// Draw the diagrams with indices `diagrams` in the SVG format in a grid on a single canvas. If specified, the
    /// grid will have `n_cols` diagrams per row, otherwise four.
    /// The result is written to the file `path`.
    pub fn draw_svg(&self, path: impl AsRef<Path>) -> Result<(), std::io::Error> {
        std::fs::write(path, self.draw_svg_range_string(.., None))?;
        Ok(())
    }

    /// Draw the diagrams with indices `diagrams` in the Typst format in a grid on a single canvas. If specified, the
    /// grid will have `n_cols` diagrams per row, otherwise four.
    /// The result is returned as a single string.
    pub fn draw_typst_string(&self, diagrams: &[usize], n_cols: Option<usize>) -> String {
        self.draw::<TypstBackend>(diagrams, n_cols)
    }

    /// Draw the diagrams with indices `diagrams` in the Typst format in a grid on a single canvas. If specified, the
    /// grid will have `n_cols` diagrams per row, otherwise four.
    /// The result is returned as a single string.
    pub fn draw_typst(&self, path: impl AsRef<Path>) -> Result<(), std::io::Error> {
        std::fs::write(path, self.draw_typst_range_string(.., None))?;
        Ok(())
    }

    /// Draw the diagrams with indices `diagrams` in the TikZ format in a grid on a single canvas. If specified, the
    /// grid will have `n_cols` diagrams per row, otherwise four.
    /// The result is returned as a single string.
    pub fn draw_tikz_string(&self, diagrams: &[usize], n_cols: Option<usize>) -> String {
        self.draw::<TikzBackend>(diagrams, n_cols)
    }

    /// Draw the diagrams with indices `diagrams` in the TikZ format in a grid on a single canvas. If specified, the
    /// grid will have `n_cols` diagrams per row, otherwise four.
    /// The result is returned as a single string.
    pub fn draw_tikz(&self, path: impl AsRef<Path>) -> Result<(), std::io::Error> {
        std::fs::write(path, self.draw_tikz_range_string(.., None))?;
        Ok(())
    }

    /// Draw the diagrams with indices in `range` in the given format on a single canvas. If specified, the
    /// grid will have `n_cols` diagrams per row, otherwise four.
    /// The result is returned as a single string.
    pub fn draw_svg_range_string(&self, range: impl RangeBounds<usize>, n_cols: Option<usize>) -> String {
        self.draw_range::<SVGBackend>(range, n_cols)
    }

    /// Draw the diagrams with indices in `range` in the given format on a single canvas. If specified, the
    /// grid will have `n_cols` diagrams per row, otherwise four.
    /// The result is returned as a single string.
    pub fn draw_typst_range_string(&self, range: impl RangeBounds<usize>, n_cols: Option<usize>) -> String {
        self.draw_range::<TypstBackend>(range, n_cols)
    }

    /// Draw the diagrams with indices in `range` in the given format on a single canvas. If specified, the
    /// grid will have `n_cols` diagrams per row, otherwise four.
    /// The result is returned as a single string.
    pub fn draw_tikz_range_string(&self, range: impl RangeBounds<usize>, n_cols: Option<usize>) -> String {
        self.draw_range::<TikzBackend>(range, n_cols)
    }

    /// Draw the diagrams with indices `diagrams` in the given format in a grid on a single canvas. If specified, the
    /// grid will have `n_cols` diagrams per row, otherwise four.
    pub fn draw<B: Backend>(&self, diagrams: &[usize], n_cols: Option<usize>) -> B::Output {
        let n_diags = diagrams.len();
        let n_cols = if let Some(n_cols) = n_cols {
            n_cols
        } else if n_diags < 4 && n_diags > 0 {
            n_diags
        } else {
            4
        };
        let n_rows = n_diags.div_ceil(n_cols);
        let mut grid = CanvasGrid::<B>::new(n_cols, n_rows);
        for (i, diag_id) in diagrams.iter().enumerate() {
            grid.draw_title(i / n_cols, i % n_cols, &format!("D{}", diagrams[i]));
            DiagramView::new(
                self.model.as_ref().unwrap(),
                &self.data[*diag_id],
                &self.momentum_labels,
            )
            .draw(grid.canvas(i / n_cols, i % n_cols));
        }
        return grid.finish();
    }

    /// Draw the diagrams with indices in `range` in the given format on a single canvas. If specified, the
    /// grid will have `n_cols` diagrams per row, otherwise four.
    pub fn draw_range<B: Backend>(&self, range: impl RangeBounds<usize>, n_cols: Option<usize>) -> B::Output {
        let min = match range.start_bound() {
            Bound::Included(n) => *n,
            Bound::Excluded(n) if *n > 0 => *n - 1,
            _ => 0,
        };
        let max = match range.end_bound() {
            Bound::Included(n) => *n,
            Bound::Excluded(n) => {
                if *n > 0 {
                    *n - 1
                } else {
                    0
                }
            }
            Bound::Unbounded => self.data.len() - 1,
        };
        let n_diags = max - min;
        let n_cols = if let Some(n_cols) = n_cols {
            n_cols
        } else if n_diags < 4 && n_diags > 0 {
            n_diags
        } else {
            4
        };
        let n_rows = n_diags.div_ceil(n_cols);
        let mut grid = CanvasGrid::<B>::new(n_cols, n_rows);
        for (i, diag_id) in (min..=max).enumerate() {
            grid.draw_title(i / n_cols, i % n_cols, &format!("D{}", diag_id));
            DiagramView::new(self.model.as_ref().unwrap(), &self.data[diag_id], &self.momentum_labels)
                .draw(grid.canvas(i / n_cols, i % n_cols));
        }
        return grid.finish();
    }
}
