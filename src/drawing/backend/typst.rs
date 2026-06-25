use super::super::style::{Color, DecorationKind};
use super::{Anchor, Backend, Decoration, Label, PathStyle, Segment, Stroke, Vec2D};

use std::f64;
use std::fmt::Write;

pub(crate) struct TypstBackend {
    buf: String,
}

impl TypstBackend {
    fn fmt_color(color: &Color) -> String {
        format!("rgb({},{},{})", color.r, color.g, color.b)
    }

    fn style(&mut self, style: &PathStyle) {
        if style.close {
            self.buf.push_str(",close:true");
        }
        write!(
            self.buf,
            r",stroke:(thickness:{:.4},paint:{}{})",
            style.width * Self::BASE_SIZE,
            Self::fmt_color(&style.color),
            match style.stroke {
                Stroke::Solid => "",
                Stroke::Dashed => r#",dash:"dashed""#,
                Stroke::Dotted => r#",dash:"dotted""#,
            }
        )
        .unwrap();
        if style.fill {
            write!(self.buf, r",fill={}", Self::fmt_color(&style.color)).unwrap();
        }
    }
}

impl Backend for TypstBackend {
    const BASE_SIZE: f64 = 10.;
    const BASE_SIZE_X: f64 = 10.;
    const BASE_SIZE_Y: f64 = 10.;

    type Output = String;

    fn particle_name(p: &crate::model::Particle) -> &str {
        &p.texname
    }

    fn init(_size_x: f64, _size_y: f64) -> Self {
        let mut buf = String::new();
        buf.push_str(
            r#"#import "@preview/cetz:0.5.2"
#import "@preview/mitex:0.2.7": *
#set page(width: auto, height: auto, margin: .5cm)
#cetz.canvas({
    import cetz.draw: *
"#,
        );
        Self { buf }
    }

    fn finish(mut self) -> Self::Output {
        self.buf.push_str("})");
        self.buf
    }

    fn draw_label(&mut self, label: Label) {
        writeln!(
            self.buf,
            r#"    content(({:.4},{:.4}),angle:{:.4}rad,anchor:"{}", [{}])"#,
            label.pos[0],
            -label.pos[1],
            label.rotate,
            match label.anchor {
                Anchor::Left => "west",
                Anchor::Center => "center",
                Anchor::Right => "east",
            },
            if label.math {
                format!(r#"#mi("{}")"#, label.text)
            } else {
                label.text
            }
        )
        .unwrap();
    }

    fn draw_segment(&mut self, start: Vec2D, segment: &Segment, style: PathStyle) {
        match segment {
            Segment::Line(end) => write!(
                self.buf,
                r#"    line(({:.4},{:.4}),({:.4},{:.4})"#,
                start[0], -start[1], end[0], -end[1]
            )
            .unwrap(),
            Segment::QuadraticBezier(c, end) => write!(
                self.buf,
                r#"    bezier(({:.4},{:.4}),({:.4},{:.4}),({:.4},{:.4})"#,
                start[0], -start[1], end[0], -end[1], c[0], -c[1]
            )
            .unwrap(),
            Segment::CubicBezier(c1, c2, end) => write!(
                self.buf,
                r#"    bezier(({:.4},{:.4}),({:.4},{:.4}),({:.4},{:.4}),({:.4},{:.4})"#,
                start[0], -start[1], end[0], -end[1], c1[0], -c1[1], c2[0], -c2[1]
            )
            .unwrap(),
        }
        self.style(&style);
        writeln!(self.buf, ")").unwrap();
        if let Some(decoration) = style.decoration {
            self.draw_decoration(segment.eval(&start, decoration.pos), decoration);
        }
    }

    fn draw_path(&mut self, start: Vec2D, curve: &Segment, segments: impl Iterator<Item = Segment>, style: PathStyle) {
        let mut current = start;
        writeln!(self.buf, "    merge-path({{").unwrap();
        for segment in segments {
            match segment {
                Segment::Line(end) => {
                    writeln!(
                        self.buf,
                        r#"        line(({:.4},{:.4}),({:.4},{:.4}))"#,
                        current[0], -current[1], end[0], -end[1]
                    )
                    .unwrap();
                    current = end;
                }
                Segment::QuadraticBezier(c, end) => {
                    writeln!(
                        self.buf,
                        r#"        bezier(({:.4},{:.4}),({:.4},{:.4}),({:.4},{:.4}))"#,
                        current[0], -current[1], end[0], -end[1], c[0], -c[1]
                    )
                    .unwrap();
                    current = end;
                }
                Segment::CubicBezier(c1, c2, end) => {
                    writeln!(
                        self.buf,
                        r#"        bezier(({:.4},{:.4}),({:.4},{:.4}),({:.4},{:.4}),({:.4},{:.4}))"#,
                        current[0], -current[1], end[0], -end[1], c1[0], -c1[1], c2[0], -c2[1]
                    )
                    .unwrap();
                    current = end;
                }
            }
        }
        write!(self.buf, "    }}").unwrap();
        self.style(&style);
        writeln!(self.buf, ")").unwrap();
        if let Some(decoration) = style.decoration {
            self.draw_decoration(curve.eval(&start, decoration.pos), decoration);
        }
    }

    fn draw_decoration(&mut self, pos: Vec2D, decoration: Decoration) {
        match decoration.kind {
            DecorationKind::Circle => writeln!(
                self.buf,
                r#"    circle(({:.4},{:.4}),radius:{:.4},fill:{},stroke:{})"#,
                pos[0],
                -pos[1],
                0.5 * decoration.size * Self::BASE_SIZE,
                Self::fmt_color(&decoration.color),
                Self::fmt_color(&decoration.color)
            )
            .unwrap(),
            DecorationKind::Triangle => {
                let v = Vec2D::from([0.35 * decoration.size * Self::BASE_SIZE, 0.]).rotate(decoration.rotate);
                let v_perp = v.perp();
                writeln!(
                    self.buf,
                    r#"    merge-path(fill:{},stroke:{},close:true, {{line(({:.4},{:.4}),({:.4},{:.4}));line((), ({:.4},{:.4}))}})"#,
                    Self::fmt_color(&decoration.color),
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
                    r#"    merge-path(stroke:{},join:false, {{
        line(({:.4},{:.4}),({:.4},{:.4}))
        line(({:.4},{:.4}),({:.4},{:.4}))
    }})"#,
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
                    r#"rect(({:.4},{:.4}),(rel: ({:.4},{:.4})),fill:{},stroke:{})"#,
                    pos[0] + v[0],
                    -pos[1] - v[1],
                    decoration.size * Self::BASE_SIZE,
                    -decoration.size * Self::BASE_SIZE,
                    Self::fmt_color(&decoration.color),
                    Self::fmt_color(&decoration.color)
                )
                .unwrap();
            }
        }
    }
}
