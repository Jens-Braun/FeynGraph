use crate::diagram::DiagramContainer;
use crate::drawing::backend::{SVGBackend, TikzBackend};
use crate::drawing::layout::{DiagramLayout, TopologyLayout};
use crate::topology::TopologyContainer;
use crate::{diagram::view::DiagramView, model::LineStyle, topology::Topology};
use backend::DrawingBackend;
use itertools::Itertools;
use layout::Layout;
use std::f64;
use std::ops::{Bound, RangeBounds};
use std::path::Path;

mod backend;
mod layout;
mod math;
mod util;

const N_SUPPORT: usize = 8;

impl Topology {
    fn draw<B: DrawingBackend>(&self, b: &mut B) {
        let layout = TopologyLayout::from(self).layout();
        b.set_scaling(&layout);
        for i in 0..self.n_external {
            let x = &layout[i];
            b.draw_label(*x, i.to_string().as_str());
        }
        for (nodes, chunk) in &self.edges.iter().chunk_by(|e| e.connected_nodes) {
            if nodes[0] == nodes[1] {
                let x1 = &layout[nodes[0]];
                let adjacent_nodes = self.nodes[nodes[0]]
                    .adjacent_nodes
                    .iter()
                    .filter(|n| **n != nodes[0])
                    .map(|n| &layout[*n])
                    .collect_vec();
                if !adjacent_nodes.is_empty() {
                    let angles = adjacent_nodes
                        .iter()
                        .map(|x2| {
                            if (x2.y - x1.y) < 0. {
                                (x2.y - x1.y).atan2(x2.x - x1.x) + 2. * f64::consts::PI
                            } else {
                                (x2.y - x1.y).atan2(x2.x - x1.x)
                            }
                        })
                        .sorted_by(|theta1, theta2| theta1.partial_cmp(theta2).unwrap())
                        .collect_vec();
                    let (n, max_angle_diff) = angles
                        .iter()
                        .circular_tuple_windows()
                        .map(|(x, y)| {
                            let delta = *y - *x;
                            if delta > 0. {
                                delta
                            } else {
                                2. * f64::consts::PI + delta
                            }
                        })
                        .enumerate()
                        .max_by(|(_, theta1), (_, theta2)| theta1.partial_cmp(theta2).unwrap())
                        .unwrap();
                    b.draw_self_loops(*x1, 1., angles[n] + max_angle_diff / 2., chunk.count());
                } else {
                    println!("{:?}", x1);
                    b.draw_self_loops(*x1, 1., -std::f64::consts::PI / 2., chunk.count());
                }
            } else {
                let x1 = &layout[nodes[0]];
                let x2 = &layout[nodes[1]];
                b.draw(*x1, *x2, &vec![(LineStyle::None, false); chunk.count()]);
            }
        }
    }

    /// Draw the topology in the TikZ (TikZiT compatible) format and write the result to the file `path`.
    pub fn draw_tikz(&self, path: impl AsRef<Path>) -> Result<(), std::io::Error> {
        let mut tikz = TikzBackend::new();
        tikz.init(1, 1);
        self.draw(&mut tikz);
        std::fs::write(path, tikz.finish())?;
        Ok(())
    }

    /// Draw the topology in the SVG format and write the result to the file `path`.
    pub fn draw_svg(&self, path: impl AsRef<Path>) -> Result<(), std::io::Error> {
        let mut svg = TikzBackend::new();
        svg.init(1, 1);
        self.draw(&mut svg);
        std::fs::write(path, svg.finish())?;
        Ok(())
    }

    /// Draw the topology in SVG format and return the result as string.
    pub fn draw_svg_string(&self) -> String {
        let mut svg = SVGBackend::new();
        svg.init(1, 1);
        self.draw(&mut svg);
        return svg.finish();
    }
}

impl TopologyContainer {
    /// Draw the topologies with indices `topologies` in SVG format in a grid on a single canvas. If specified, the
    /// grid will have `n_cols` topologies per row, otherwise four.
    pub fn draw_svg(&self, topologies: &[usize], n_cols: Option<usize>) -> String {
        let n_topos = topologies.len();
        let n_cols = if let Some(n_cols) = n_cols {
            n_cols
        } else if n_topos < 4 {
            n_topos
        } else {
            4
        };
        let n_rows = n_topos.div_ceil(n_cols);
        let mut svg = SVGBackend::new();
        svg.init(n_rows, n_cols);
        for (i, topo_id) in topologies.iter().enumerate() {
            svg.init_group(i / n_cols, i % n_cols);
            self.data[*topo_id].draw(&mut svg);
            svg.draw_label_raw([150., 20.].into(), &format!("T{}", topo_id));
            svg.finish_group();
        }
        return svg.finish();
    }

    /// Draw the topologies with indices in `range` in SVG format on a single canvas. If specified, the
    /// grid will have `n_cols` topologies per row, otherwise four.
    pub fn draw_svg_range(&self, range: impl RangeBounds<usize>, n_cols: Option<usize>) -> String {
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
        let mut svg = SVGBackend::new();
        svg.init(n_rows, n_cols);
        for (i, topo_id) in (min..=max).enumerate() {
            svg.init_group(i / n_cols, i % n_cols);
            self.data[topo_id].draw(&mut svg);
            svg.draw_label_raw([150., 20.].into(), &format!("T{}", topo_id));
            svg.finish_group();
        }
        return svg.finish();
    }
}

impl DiagramView<'_> {
    fn draw<B: DrawingBackend>(&self, b: &mut B) {
        let layout = DiagramLayout::from(self).layout();
        b.set_scaling(&layout);
        let n_ext = self.n_ext();
        for l in self.incoming().chain(self.outgoing()) {
            let x1 = &layout[l.leg_index];
            let x2 = &layout[l.leg.vertex + n_ext];
            b.draw_label(*x1, &format!("{}({})", l.particle().name, l.leg_index));
            b.draw(*x1, *x2, &[(l.particle().linestyle.clone(), l.particle().is_anti())]);
        }
        for (vertices, chunk) in &self.propagators().chunk_by(|p| p.propagator.vertices) {
            let x1 = &layout[vertices[0] + n_ext];
            let x2 = &layout[vertices[1] + n_ext];
            let styles = chunk
                .into_iter()
                .map(|p| (p.particle().linestyle.clone(), p.particle().is_anti()))
                .collect_vec();
            b.draw(*x1, *x2, &styles);
        }
    }

    /// Draw the diagram in the TikZ (TikZiT compatible) format and write the result to the file `path`.
    pub fn draw_tikz(&self, path: impl AsRef<Path>) -> Result<(), std::io::Error> {
        let mut tikz = TikzBackend::new();
        tikz.init(1, 1);
        self.draw(&mut tikz);
        std::fs::write(path, tikz.finish())?;
        Ok(())
    }

    /// Draw the topology in the SVG format and return the result as string.
    pub fn draw_svg_str(&self) -> String {
        let mut svg = SVGBackend::new();
        svg.init(1, 1);
        self.draw(&mut svg);
        return svg.finish();
    }

    /// Draw the topology in the SVG format and write the result to the file `path`.
    pub fn draw_svg(&self, path: impl AsRef<Path>) -> Result<(), std::io::Error> {
        let mut svg = SVGBackend::new();
        svg.init(1, 1);
        self.draw(&mut svg);
        std::fs::write(path, svg.finish())?;
        Ok(())
    }
}

impl DiagramContainer {
    /// Draw the diagrams with indices `diagrams` in SVG format in a grid on a single canvas. If specified, the
    /// grid will have `n_cols` diagrams per row, otherwise four.
    pub fn draw_svg(&self, diagrams: &[usize], n_cols: Option<usize>) -> String {
        let n_diags = diagrams.len();
        let n_cols = if let Some(n_cols) = n_cols {
            n_cols
        } else if n_diags < 4 {
            n_diags
        } else {
            4
        };
        let n_rows = n_diags.div_ceil(n_cols);
        let mut svg = SVGBackend::new();
        svg.init(n_rows, n_cols);
        for (i, diag_id) in diagrams.iter().enumerate() {
            svg.init_group(i / n_cols, i % n_cols);
            DiagramView::new(
                self.model.as_ref().unwrap(),
                &self.data[*diag_id],
                &self.momentum_labels,
            )
            .draw(&mut svg);
            svg.draw_label_raw([150., 20.].into(), &format!("D{}", diagrams[i]));
            svg.finish_group();
        }
        return svg.finish();
    }

    /// Draw the diagrams with indices in `range` in SVG format on a single canvas. If specified, the
    /// grid will have `n_cols` diagrams per row, otherwise four.
    pub fn draw_svg_range(&self, range: impl RangeBounds<usize>, n_cols: Option<usize>) -> String {
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
        let n_diags = max - min;
        let n_cols = if let Some(n_cols) = n_cols {
            n_cols
        } else if n_diags < 4 {
            n_diags
        } else {
            4
        };
        let n_rows = n_diags.div_ceil(n_cols);
        let mut svg = SVGBackend::new();
        svg.init(n_rows, n_cols);
        for (i, diag_id) in (min..=max).enumerate() {
            svg.init_group(i / n_cols, i % n_cols);
            DiagramView::new(self.model.as_ref().unwrap(), &self.data[diag_id], &self.momentum_labels).draw(&mut svg);
            svg.draw_label_raw([150., 20.].into(), &format!("D{}", diag_id));
            svg.finish_group();
        }
        return svg.finish();
    }
}
