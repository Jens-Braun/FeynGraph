use feyngraph_core::model::LineStyle;

use super::{
    components::{Label, Path, Segment},
    style::{Anchor, PathStyle},
};

use crate::{Decoration, backend::Backend, math::Vec2D, style::Theme};

pub struct CanvasGrid<B: Backend> {
    backend: B,
    cols: usize,
    rows: usize,
}

impl<B: Backend> CanvasGrid<B> {
    pub fn new(cols: usize, rows: usize) -> Self {
        let theme = Theme::get_global();
        return Self {
            cols,
            rows,
            backend: B::init(
                B::BASE_SIZE_X * cols as f64 * (1. + 2. * theme.border_size),
                B::BASE_SIZE_Y * rows as f64 * (1. + theme.title_size + theme.border_size),
            ),
        };
    }

    pub fn finish(self) -> B::Output {
        return self.backend.finish();
    }

    pub fn draw_title(&mut self, col: usize, row: usize, label: &str) {
        let theme = Theme::get_global();
        self.backend.draw_label(Label {
            pos: Vec2D {
                x: ((1. + 2. * theme.border_size) * row as f64 + theme.border_size + 0.5) * B::BASE_SIZE_X,
                y: (col as f64 * (1. + theme.title_size + theme.border_size) + 0.7 * theme.title_size) * B::BASE_SIZE_Y,
            },
            rotate: 0.,
            anchor: Anchor::Center,
            math: false,
            text: label.to_owned(),
        });
    }

    pub fn canvas<'a>(&'a mut self, col: usize, row: usize) -> Canvas<'a, B> {
        assert!(col <= self.cols);
        assert!(row <= self.rows);
        let theme = Theme::get_global();
        Canvas {
            backend: &mut self.backend,
            origin: Vec2D::from([
                ((1. + 2. * theme.border_size) * row as f64 + theme.border_size) * B::BASE_SIZE_X,
                (theme.title_size + col as f64 * (1. + theme.title_size + theme.border_size)) * B::BASE_SIZE_Y,
            ]),
            labels: Vec::new(),
            paths: Vec::new(),
            decorations: Vec::new(),
        }
    }
}

pub struct Canvas<'a, B: Backend> {
    backend: &'a mut B,
    origin: Vec2D,
    labels: Vec<Label>,
    paths: Vec<Path>,
    decorations: Vec<(Vec2D, Decoration)>,
}

impl<'a, B: Backend> Canvas<'a, B> {
    pub fn push_label(&mut self, text: String, pos: Vec2D, anchor: Anchor, math: bool) {
        self.labels.push(Label {
            pos,
            rotate: 0.,
            anchor,
            math,
            text,
        });
    }

    pub fn push_path(
        &mut self,
        start: Vec2D,
        segment: Segment,
        style: PathStyle,
        linetype: LineStyle,
        label: Option<String>,
    ) {
        self.paths.push(Path {
            start,
            segment,
            style,
            linetype,
            label,
        });
    }

    pub fn push_decoration(&mut self, pos: Vec2D, d: Decoration) {
        self.decorations.push((pos, d));
    }

    fn bounding_box(&self) -> (Vec2D, Vec2D) {
        let (path_min, path_max) = self
            .paths
            .iter()
            .map(|p| p.bounding_box())
            .fold(None, |acc: Option<(Vec2D, Vec2D)>, (min, max)| {
                if let Some(acc) = acc {
                    Some((
                        Vec2D::from([acc.0[0].min(min[0]), acc.0[1].min(min[1])]),
                        Vec2D::from([acc.1[0].max(max[0]), acc.1[1].max(max[1])]),
                    ))
                } else {
                    Some((min, max))
                }
            })
            .unwrap_or((Vec2D::from([0., 0.]), Vec2D::from([1., 1.])));
        let base_size = (path_max - path_min).norm();
        let (label_min, label_max) = self
            .labels
            .iter()
            .map(|l| l.bounding_box(base_size))
            .fold(None, |acc: Option<(Vec2D, Vec2D)>, (min, max)| {
                if let Some(acc) = acc {
                    Some((
                        Vec2D::from([acc.0[0].min(min[0]), acc.0[1].min(min[1])]),
                        Vec2D::from([acc.1[0].max(max[0]), acc.1[1].max(max[1])]),
                    ))
                } else {
                    Some((min, max))
                }
            })
            .unwrap_or((Vec2D::from([0., 0.]), Vec2D::from([1., 1.])));
        (
            Vec2D::from([path_min[0].min(label_min[0]), path_min[1].min(label_min[1])]),
            Vec2D::from([path_max[0].max(label_max[0]), path_max[1].max(label_max[1])]),
        )
    }

    pub fn finish(self) {
        let (v_min, v_max) = self.bounding_box();
        let v_diff = v_max - v_min;
        let scale = (B::BASE_SIZE_X / v_diff[0]).min(B::BASE_SIZE_Y / v_diff[1]);
        for mut path in self.paths {
            path.shift(-v_min);
            path.scale(scale, scale);
            path.shift(self.origin + 0.5 * (Vec2D::from([B::BASE_SIZE_X, B::BASE_SIZE_Y]) - scale * (v_max - v_min)));
            path.draw(self.backend);
        }
        for (mut pos, d) in self.decorations {
            pos = pos - v_min;
            pos.scale(scale, scale);
            pos = pos + self.origin + 0.5 * (Vec2D::from([B::BASE_SIZE_X, B::BASE_SIZE_Y]) - scale * (v_max - v_min));
            self.backend.draw_decoration(pos, d);
        }
        for mut label in self.labels {
            label.shift(-v_min);
            label.scale(scale, scale);
            label.shift(self.origin + 0.5 * (Vec2D::from([B::BASE_SIZE_X, B::BASE_SIZE_Y]) - scale * (v_max - v_min)));
            self.backend.draw_label(label);
        }
    }
}
