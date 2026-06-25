use itertools::Itertools;

use super::{
    backend::Backend,
    consts::*,
    math::Vec2D,
    style::{Anchor, Decoration, DecorationKind, PathStyle, Stroke},
};
use crate::{
    drawing::{Color, style::Theme},
    model::LineStyle,
};

pub struct Label {
    pub pos: Vec2D,
    pub rotate: f64,
    pub anchor: Anchor,
    pub math: bool,
    pub text: String,
}

impl Label {
    pub(crate) fn shift(&mut self, v: Vec2D) {
        self.pos = self.pos + v;
    }

    pub(crate) fn scale(&mut self, scale_x: f64, scale_y: f64) {
        self.pos.scale(scale_x, scale_y);
    }

    pub(crate) fn bounding_box(&self, base_size: f64) -> (Vec2D, Vec2D) {
        let width = base_size * self.text.chars().count() as f64 * Theme::get_global().character_width;
        let height = base_size * Theme::get_global().font_size;
        match self.anchor {
            Anchor::Left => (self.pos, Vec2D::from([self.pos[0] + width, self.pos[1] - height])),
            Anchor::Center => (
                Vec2D::from([self.pos[0] - 0.5 * width, self.pos[1]]),
                Vec2D::from([self.pos[0] + 0.5 * width, self.pos[1] - height]),
            ),
            Anchor::Right => (
                Vec2D::from([self.pos[0] - width, self.pos[1]]),
                Vec2D::from([self.pos[0], self.pos[1] - height]),
            ),
        }
    }
}

pub struct Path {
    pub start: Vec2D,
    pub segment: Segment,
    pub style: PathStyle,
    pub linetype: LineStyle,
    pub label: Option<String>,
}

impl Path {
    pub(crate) fn eval(&self, t: f64) -> Vec2D {
        self.segment.eval(&self.start, t)
    }

    pub(crate) fn unit_tangent(&self, t: f64) -> Vec2D {
        self.segment.unit_tangent(&self.start, t)
    }

    pub(crate) fn arclen(&self) -> f64 {
        (0..ARCLEN_RESOLUTION)
            .into_iter()
            .map(|i| self.eval(i as f64 / (ARCLEN_RESOLUTION - 1) as f64))
            .tuple_windows()
            .map(|(x1, x2)| (x2 - x1).norm())
            .sum::<f64>()
    }

    pub(crate) fn arclen_parameterization(&self, l: f64) -> f64 {
        let mut current_len = 0.;
        let mut current_vec = self.start;
        for i in 1..ARCLEN_RESOLUTION {
            let next = self.eval(i as f64 / (ARCLEN_RESOLUTION - 1) as f64);
            let dl = (next - current_vec).norm();
            if current_len + dl >= l {
                return f64::clamp(
                    (i as f64 + (l - current_len - dl) / dl) / (ARCLEN_RESOLUTION - 1) as f64,
                    0.,
                    1.,
                );
            } else {
                current_vec = next;
                current_len += dl;
            }
        }
        1.
    }

    pub(crate) fn raw_supports(&self, pattern_count: usize) -> Vec<Vec2D> {
        let segment_len = self.arclen() / (pattern_count - 1) as f64;
        (0..pattern_count)
            .map(|i| Vec2D::from([segment_len * i as f64, 0.]))
            .collect()
    }

    pub(crate) fn transform_along(&self, x: Vec2D) -> Vec2D {
        let t = self.arclen_parameterization(x[0]);
        let v = self.eval(t);
        let normal = self.unit_tangent(t).perp();
        return v + x[1] * normal;
    }

    pub(crate) fn bounding_box(&self) -> (Vec2D, Vec2D) {
        self.segment.bounding_box(&self.start)
    }

    pub(crate) fn scale(&mut self, scale_x: f64, scale_y: f64) {
        self.start.scale(scale_x, scale_y);
        self.segment.scale(scale_x, scale_y);
    }

    pub(crate) fn shift(&mut self, v: Vec2D) {
        self.start = self.start + v;
        self.segment.shift(v);
    }

    fn self_loop(&self) -> bool {
        match self.segment {
            Segment::Line(x2) | Segment::QuadraticBezier(_, x2) | Segment::CubicBezier(.., x2) => self.start == x2,
        }
    }

    pub(crate) fn draw<B: Backend>(&self, b: &mut B) {
        let theme = Theme::get_global();
        let pattern_count = theme.pattern_count
            * if self.self_loop() {
                theme.self_loop_multiplier
            } else {
                1
            };
        let self_loop_reduction = if self.self_loop() {
            theme.self_loop_reduction
        } else {
            1.
        };
        match self.linetype {
            LineStyle::None => {
                b.draw_segment(self.start, &self.segment, self.style);
            }
            LineStyle::Dashed => {
                b.draw_segment(self.start, &self.segment, self.style.stroke(Stroke::Dashed));
            }
            LineStyle::Dotted => {
                b.draw_segment(self.start, &self.segment, self.style.stroke(Stroke::Dotted));
            }
            LineStyle::Straight => {
                b.draw_segment(
                    self.start,
                    &self.segment,
                    self.style.decoration(Decoration {
                        color: self.style.color,
                        kind: DecorationKind::Triangle,
                        pos: 0.5,
                        rotate: self.unit_tangent(0.5).arg(),
                        size: theme.decoration_size,
                    }),
                );
            }
            LineStyle::Wavy => {
                self.draw_wavy(b, pattern_count, self_loop_reduction, &theme);
            }
            LineStyle::Curly => {
                self.draw_curly(b, pattern_count, self_loop_reduction, &theme);
            }
            LineStyle::Swavy => {
                self.draw_wavy(b, pattern_count, self_loop_reduction, &theme);
                b.draw_segment(self.start, &self.segment, self.style);
            }
            LineStyle::Scurly => {
                self.draw_curly(b, pattern_count, self_loop_reduction, &theme);
                b.draw_segment(self.start, &self.segment, self.style);
            }
            LineStyle::Double => {
                b.draw_segment(self.start, &self.segment, self.style.width(self.style.width * 2.5));
                b.draw_segment(
                    self.start,
                    &self.segment,
                    self.style.width(self.style.width * 0.75).color(Color::WHITE),
                );
            }
        }
        if let Some(ref label) = self.label
            && theme.label_propagators
        {
            b.draw_label(Label {
                pos: self.eval(0.5) + theme.propagator_label_distance * B::BASE_SIZE * self.unit_tangent(0.5).perp(),
                rotate: 0.,
                anchor: Anchor::Center,
                math: true,
                text: label.clone(),
            });
        }
    }

    fn draw_wavy<B: Backend>(&self, b: &mut B, pattern_count: usize, self_loop_reduction: f64, theme: &Theme) {
        b.draw_path(
            self.start,
            &self.segment,
            self.raw_supports(pattern_count)
                .into_iter()
                .tuple_windows()
                .map(|(s1, s2)| {
                    let v = s2 - s1;
                    let v_p = self_loop_reduction * theme.wavy_amplitude * B::BASE_SIZE * v.normalize().perp();
                    Segment::CubicBezier(
                        self.transform_along(s1 + 0.5 * v + v_p),
                        self.transform_along(s1 + 0.5 * v - v_p),
                        self.transform_along(s2),
                    )
                }),
            self.style,
        );
    }

    fn draw_curly<B: Backend>(&self, b: &mut B, pattern_count: usize, self_loop_reduction: f64, theme: &Theme) {
        b.draw_path(
            self.start,
            &self.segment,
            self.raw_supports(pattern_count)
                .into_iter()
                .tuple_windows()
                .enumerate()
                .flat_map(|(i, (xi, xf))| {
                    let v = xf - xi;
                    let d = v.norm();
                    let s1 = xi + d * CURL_SUPPORT[0];
                    let c11 = xi + self_loop_reduction * theme.curly_amplitude * B::BASE_SIZE * CURL_CONTROL[0];
                    let c12 = s1 + self_loop_reduction * theme.curly_amplitude * B::BASE_SIZE * CURL_CONTROL[1];
                    if i + 2 == pattern_count {
                        vec![Segment::CubicBezier(
                            self.transform_along(c11),
                            self.transform_along(c12),
                            self.transform_along(xf),
                        )]
                    } else {
                        let s2 = xi + d * CURL_SUPPORT[1];
                        let c21 = s1 + self_loop_reduction * theme.curly_amplitude * B::BASE_SIZE * CURL_CONTROL[2];
                        let c22 = s2 + self_loop_reduction * theme.curly_amplitude * B::BASE_SIZE * CURL_CONTROL[3];
                        vec![
                            Segment::CubicBezier(
                                self.transform_along(c11),
                                self.transform_along(c12),
                                self.transform_along(s1),
                            ),
                            Segment::CubicBezier(
                                self.transform_along(c21),
                                self.transform_along(c22),
                                self.transform_along(s2),
                            ),
                        ]
                    }
                }),
            self.style,
        );
    }
}

pub enum Segment {
    Line(Vec2D),
    QuadraticBezier(Vec2D, Vec2D),
    CubicBezier(Vec2D, Vec2D, Vec2D),
}

impl Segment {
    pub(crate) fn eval(&self, x1: &Vec2D, t: f64) -> Vec2D {
        if t < 0. || t > 1. {
            return Vec2D::from([f64::NAN, f64::NAN]);
        }
        match self {
            Self::Line(x2) => {
                return (1. - t) * *x1 + t * *x2;
            }
            Self::QuadraticBezier(c, x2) => {
                let tp = 1. - t;
                return tp * (tp * *x1 + t * *c) + t * (tp * *c + t * *x2);
            }
            Self::CubicBezier(c1, c2, x2) => {
                let tp = 1. - t;
                return tp.powi(3) * *x1 + 3. * tp * tp * t * *c1 + 3. * tp * t * t * *c2 + t.powi(3) * *x2;
            }
        }
    }

    fn bounding_box(&self, x1: &Vec2D) -> (Vec2D, Vec2D) {
        match self {
            Self::Line(x2) => {
                let (xmin, xmax) = if x1[0] < x2[0] { (x1[0], x2[0]) } else { (x2[0], x1[0]) };
                let (ymin, ymax) = if x1[1] < x2[1] { (x1[1], x2[1]) } else { (x2[1], x1[1]) };
                (Vec2D::from([xmin, ymin]), Vec2D::from([xmax, ymax]))
            }
            Self::QuadraticBezier(c, x2) => {
                let tx = (c[0] - x1[0]) / (2. * c[0] - x1[0] - x2[0]);
                let ty = (c[1] - x1[1]) / (2. * c[1] - x1[1] - x2[1]);
                let x = self.eval(x1, tx)[0];
                let y = self.eval(x1, ty)[1];
                let xmin = [x, x1[0], x2[0]].into_iter().reduce(f64::min).unwrap();
                let xmax = [x, x1[0], x2[0]].into_iter().reduce(f64::max).unwrap();
                let ymin = [y, x1[1], x2[1]].into_iter().reduce(f64::min).unwrap();
                let ymax = [y, x1[1], x2[1]].into_iter().reduce(f64::max).unwrap();
                (Vec2D::from([xmin, ymin]), Vec2D::from([xmax, ymax]))
            }
            Self::CubicBezier(c1, c2, x2) => {
                fn t_ext(x1: f64, c1: f64, c2: f64, x2: f64) -> (f64, f64) {
                    let denom = (3. * (c1 - c2) - x1 + x2).recip();
                    let sqrt = f64::sqrt((c1 - c2) * (c1 - c2) + c1 * c2 - c2 * x1 - c1 * x2 + x1 * x2);
                    let num1 = 2. * c1 - c2 - x1;
                    ((num1 + sqrt) * denom, (num1 - sqrt) * denom)
                }
                let (tx1, tx2) = t_ext(x1[0], c1[0], c2[0], x2[0]);
                let (xe1, xe2) = (self.eval(x1, tx1)[0], self.eval(x1, tx2)[0]);
                let (ty1, ty2) = t_ext(x1[1], c1[1], c2[1], x2[1]);
                let (ye1, ye2) = (self.eval(x1, ty1)[1], self.eval(x1, ty2)[1]);
                let xmin = [xe1, xe2, x1[0], x2[0]].into_iter().reduce(f64::min).unwrap();
                let xmax = [xe1, xe2, x1[0], x2[0]].into_iter().reduce(f64::max).unwrap();
                let ymin = [ye1, ye2, x1[1], x2[1]].into_iter().reduce(f64::min).unwrap();
                let ymax = [ye1, ye2, x1[1], x2[1]].into_iter().reduce(f64::max).unwrap();
                (Vec2D::from([xmin, ymin]), Vec2D::from([xmax, ymax]))
            }
        }
    }

    fn scale(&mut self, scale_x: f64, scale_y: f64) {
        match self {
            Self::Line(x2) => x2.scale(scale_x, scale_y),
            Self::QuadraticBezier(c, x2) => {
                c.scale(scale_x, scale_y);
                x2.scale(scale_x, scale_y);
            }
            Self::CubicBezier(c1, c2, x2) => {
                c1.scale(scale_x, scale_y);
                c2.scale(scale_x, scale_y);
                x2.scale(scale_x, scale_y);
            }
        }
    }

    fn shift(&mut self, v: Vec2D) {
        match self {
            Self::Line(x2) => *x2 = *x2 + v,
            Self::QuadraticBezier(c, x2) => {
                *c = *c + v;
                *x2 = *x2 + v;
            }
            Self::CubicBezier(c1, c2, x2) => {
                *c1 = *c1 + v;
                *c2 = *c2 + v;
                *x2 = *x2 + v;
            }
        }
    }

    fn midpoint(&self, x1: &Vec2D) -> Vec2D {
        self.eval(x1, 0.5)
    }

    fn mid_tangent(&self, x1: &Vec2D) -> Vec2D {
        match self {
            Self::Line(x2) | Self::QuadraticBezier(_, x2) => (*x2 - *x1).normalize(),
            Self::CubicBezier(c1, c2, x2) => (*x1 + *c1 - *c2 - *x2).normalize(),
        }
    }

    fn unit_tangent(&self, x1: &Vec2D, t: f64) -> Vec2D {
        if t < 0. || t > 1. {
            return Vec2D::from([f64::NAN, f64::NAN]);
        }
        match self {
            Self::Line(x2) => {
                return (*x2 - *x1).normalize();
            }
            Self::QuadraticBezier(c, x2) => {
                let tp = 1. - t;
                return (*c - 2. * t * *c - tp * *x1 + t * *x2).normalize();
            }
            Self::CubicBezier(c1, c2, x2) => {
                let tp = 1. - t;
                return (*c2 * (2. - 3. * t) * t + *c1 * tp * (1. - 3. * t) - tp * tp * *x1 + t * t * *x2).normalize();
            }
        }
    }

    pub(crate) fn invert(&mut self, start: Vec2D) -> Vec2D {
        match self {
            Self::Line(end) => std::mem::replace(end, start),
            Self::QuadraticBezier(_, end) => std::mem::replace(end, start),
            Self::CubicBezier(c1, c2, end) => {
                std::mem::swap(c1, c2);
                std::mem::replace(end, start)
            }
        }
    }
}
