use super::super::style::{Color, DecorationKind};
use super::{Anchor, Backend, Decoration, Label, PathStyle, Segment, Stroke, Vec2D};

use std::f64;
use std::fmt::Write;

pub(crate) struct TikzBackend {
    buf: String,
}

impl TikzBackend {
    fn fmt_color(color: &Color) -> String {
        format!("{{rgb,255:red,{};green,{};blue,{}}}", color.r, color.g, color.b)
    }

    fn style(&mut self, style: &PathStyle) {
        write!(
            self.buf,
            r"line width={:.4}cm,color={}{}",
            0.5 * style.width * Self::BASE_SIZE,
            Self::fmt_color(&style.color),
            match style.stroke {
                Stroke::Solid => "",
                Stroke::Dashed => r#",dashed"#,
                Stroke::Dotted => r#",dotted"#,
            }
        )
        .unwrap();
        if style.fill {
            write!(self.buf, r",fill={}", Self::fmt_color(&style.color)).unwrap();
        }
    }
}

impl Backend for TikzBackend {
    const BASE_SIZE: f64 = 10.;
    const BASE_SIZE_X: f64 = 10.;
    const BASE_SIZE_Y: f64 = 10.;
    type Output = String;

    fn particle_name(p: &crate::model::Particle) -> &str {
        &p.texname
    }

    fn init(_size_x: f64, _size_y: f64) -> Self {
        let mut buf = String::new();
        buf.push_str("\\begin{tikzpicture}\n");
        Self { buf }
    }

    fn finish(mut self) -> Self::Output {
        self.buf.push_str(r"\end{tikzpicture}");
        self.buf
    }

    fn draw_label(&mut self, label: Label) {
        writeln!(
            self.buf,
            r#"    \draw ({:.4},{:.4}) node[anchor={},rotate={}] {{{}}};"#,
            label.pos[0],
            -label.pos[1],
            match label.anchor {
                Anchor::Left => "west",
                Anchor::Center => "center",
                Anchor::Right => "east",
            },
            label.rotate,
            if label.math {
                format!("${}$", label.text)
            } else {
                label.text
            }
        )
        .unwrap();
    }

    fn draw_segment(&mut self, start: Vec2D, segment: &Segment, style: PathStyle) {
        self.buf.push_str(r"    \draw[");
        self.style(&style);
        self.buf.push_str("]");
        match segment {
            Segment::Line(end) => write!(
                self.buf,
                r#" ({:.4},{:.4})--({:.4},{:.4})"#,
                start[0], -start[1], end[0], -end[1]
            )
            .unwrap(),
            Segment::QuadraticBezier(c, end) => write!(
                self.buf,
                r#" ({:.4},{:.4}).. controls ({:.4},{:.4}) .. ({:.4},{:.4})"#,
                start[0], -start[1], c[0], -c[1], end[0], -end[1],
            )
            .unwrap(),
            Segment::CubicBezier(c1, c2, end) => write!(
                self.buf,
                r#" ({:.4},{:.4}).. controls ({:.4},{:.4}) and ({:.4},{:.4}) .. ({:.4},{:.4})"#,
                start[0], -start[1], c1[0], -c1[1], c2[0], -c2[1], end[0], -end[1]
            )
            .unwrap(),
        }
        if style.close {
            self.buf.push_str("-- cycle");
        }
        self.buf.push_str(";\n");
        if let Some(decoration) = style.decoration {
            self.draw_decoration(segment.eval(&start, decoration.pos), decoration);
        }
    }

    fn draw_path(&mut self, start: Vec2D, curve: &Segment, segments: impl Iterator<Item = Segment>, style: PathStyle) {
        self.buf.push_str(r"    \draw[");
        self.style(&style);
        self.buf.push_str("]");
        write!(self.buf, r" ({:.4},{:.4})", start[0], -start[1]).unwrap();
        for segment in segments {
            match segment {
                Segment::Line(end) => write!(self.buf, r#" --({:.4},{:.4})"#, end[0], -end[1]).unwrap(),
                Segment::QuadraticBezier(c, end) => write!(
                    self.buf,
                    r#" .. controls ({:.4},{:.4}) .. ({:.4},{:.4})"#,
                    c[0], -c[1], end[0], -end[1],
                )
                .unwrap(),
                Segment::CubicBezier(c1, c2, end) => write!(
                    self.buf,
                    r#" .. controls ({:.4},{:.4}) and ({:.4},{:.4}) .. ({:.4},{:.4})"#,
                    c1[0], -c1[1], c2[0], -c2[1], end[0], -end[1],
                )
                .unwrap(),
            }
        }
        if style.close {
            self.buf.push_str("-- cycle");
        }
        self.buf.push_str(";\n");
        if let Some(decoration) = style.decoration {
            self.draw_decoration(curve.eval(&start, decoration.pos), decoration);
        }
    }

    fn draw_decoration(&mut self, pos: Vec2D, decoration: Decoration) {
        match decoration.kind {
            DecorationKind::Circle => writeln!(
                self.buf,
                r#"    \filldraw[color={}] ({:.4},{:.4}) circle ({:.4});"#,
                Self::fmt_color(&decoration.color),
                pos[0],
                -pos[1],
                0.5 * decoration.size * Self::BASE_SIZE,
            )
            .unwrap(),
            DecorationKind::Triangle => {
                let v = Vec2D::from([0.35 * decoration.size * Self::BASE_SIZE, 0.]).rotate(decoration.rotate);
                let v_perp = v.perp();
                writeln!(
                    self.buf,
                    r#"    \filldraw[color={}] ({:.4},{:.4})--({:.4},{:.4})--({:.4},{:.4})--cycle;"#,
                    Self::fmt_color(&decoration.color),
                    pos[0] + v[0],
                    -pos[1] - v[1],
                    pos[0] - v[0] + v_perp[0],
                    -pos[1] + v[1] - v_perp[1],
                    pos[0] - v[0] - v_perp[0],
                    -pos[1] + v[1] + v_perp[1],
                )
                .unwrap();
            }
            DecorationKind::Cross => {
                let v = f64::consts::FRAC_1_SQRT_2
                    * decoration.size
                    * Self::BASE_SIZE
                    * Vec2D::from([1., 1.]).rotate(decoration.rotate);
                writeln!(
                    self.buf,
                    r#"    \draw[color={}] ({:.4},{:.4})--({:.4},{:.4}) ({:.4},{:.4})--({:.4},{:.4});"#,
                    Self::fmt_color(&decoration.color),
                    pos[0] + v[0],
                    -pos[1] - v[1],
                    pos[0] - v[0],
                    -pos[1] + v[1],
                    pos[0] - v[0],
                    -pos[1] - v[1],
                    pos[0] + v[0],
                    -pos[1] + v[1],
                )
                .unwrap();
            }
            DecorationKind::Box => {
                let v = f64::consts::FRAC_1_SQRT_2
                    * decoration.size
                    * Self::BASE_SIZE
                    * Vec2D::from([-0.5, -0.5]).rotate(decoration.rotate);
                writeln!(
                    self.buf,
                    r#"    \filldraw[color={}] ({:.4},{:.4}) rectangle ({:.4},{:.4});"#,
                    Self::fmt_color(&decoration.color),
                    pos[0] + v[0],
                    -pos[1] - v[1],
                    pos[0] - v[0],
                    -pos[1] + v[1]
                )
                .unwrap();
            }
        }
    }
}
