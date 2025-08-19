#![allow(unused_variables)]
use crate::drawing::layout::NodeLayout;
use crate::drawing::math::{Vec2D, cubic_bezier, support_points};
use crate::model::LineStyle;
use std::f64::consts::PI;
use std::fmt::Write;

/// Support and control points for a cubic Bezier approximation of the parametric curve
/// $$\begin{pmatrix} \left(2 + \frac{3}{4} t - 2 \cos t\right) / \left(2 + \frac{9 \pi}{4}\right) \\ -\frac{2}{2} \sin t\end{pmatrix}$$
const CURL_SUPPORT: [Vec2D; 2] = [
    Vec2D {
        x: (32. + 6. * PI) / (16. + 9. * PI),
        y: 0.,
    },
    Vec2D {
        x: 12. * PI / (16. + 9. * PI),
        y: 0.,
    },
];
const CURL_CONTROL: [Vec2D; 4] = [
    Vec2D {
        x: 8. / (16. + 9. * PI),
        y: -2. / 3.,
    },
    Vec2D {
        x: -8. / (16. + 9. * PI),
        y: -2. / 3.,
    },
    Vec2D {
        x: 8. / (16. + 9. * PI),
        y: 2. / 3.,
    },
    Vec2D {
        x: -8. / (16. + 9. * PI),
        y: 2. / 3.,
    },
];
const CURL_AMPLITUDE: f64 = 20.;

pub(crate) trait DrawingBackend {
    fn init(&mut self, n_rows: usize, n_cols: usize);
    fn finish(&mut self) -> String;

    fn init_group(&mut self, row: usize, col: usize);
    fn finish_group(&mut self);

    fn draw_label(&mut self, x: Vec2D, label: &str);
    fn draw_self_loops(&mut self, x: Vec2D, r: f64, theta: f64, n_lines: usize);
    fn draw_line(&mut self, x1: Vec2D, c1: Option<Vec2D>, c2: Option<Vec2D>, x2: Vec2D);
    fn draw_dashed(&mut self, x1: Vec2D, c1: Option<Vec2D>, c2: Option<Vec2D>, x2: Vec2D);
    fn draw_dotted(&mut self, x1: Vec2D, c1: Option<Vec2D>, c2: Option<Vec2D>, x2: Vec2D);
    fn draw_straight(&mut self, x1: Vec2D, c1: Option<Vec2D>, c2: Option<Vec2D>, x2: Vec2D);
    fn draw_wavy(&mut self, x1: Vec2D, c1: Option<Vec2D>, c2: Option<Vec2D>, x2: Vec2D);
    fn draw_curly(&mut self, x1: Vec2D, c1: Option<Vec2D>, c2: Option<Vec2D>, x2: Vec2D);
    fn draw_scurly(&mut self, x1: Vec2D, c1: Option<Vec2D>, c2: Option<Vec2D>, x2: Vec2D);
    fn draw_swavy(&mut self, x1: Vec2D, c1: Option<Vec2D>, c2: Option<Vec2D>, x2: Vec2D);
    fn draw_double(&mut self, x1: Vec2D, c1: Option<Vec2D>, c2: Option<Vec2D>, x2: Vec2D);

    fn draw_style(&mut self, x1: Vec2D, c1: Option<Vec2D>, c2: Option<Vec2D>, x2: Vec2D, style: &LineStyle) {
        match *style {
            LineStyle::Dashed => self.draw_dashed(x1, c1, c2, x2),
            LineStyle::Dotted => self.draw_dotted(x1, c1, c2, x2),
            LineStyle::Straight => self.draw_straight(x1, c1, c2, x2),
            LineStyle::Wavy => self.draw_wavy(x1, c1, c2, x2),
            LineStyle::Curly => self.draw_curly(x1, c1, c2, x2),
            LineStyle::Scurly => self.draw_scurly(x1, c1, c2, x2),
            LineStyle::Swavy => self.draw_swavy(x1, c1, c2, x2),
            LineStyle::Double => self.draw_double(x1, c1, c2, x2),
            LineStyle::None => self.draw_line(x1, c1, c2, x2),
        }
    }

    fn draw(&mut self, x1: Vec2D, x2: Vec2D, styles: &[(LineStyle, bool)]) {
        match styles.len() {
            1 => {
                if styles[0].1 {
                    self.draw_style(x2, None, None, x1, &styles[0].0);
                } else {
                    self.draw_style(x1, None, None, x2, &styles[0].0);
                }
            }
            n => {
                let v_conn = (x2 - x1) / 2.;
                let v_perp = Vec2D::from([-v_conn.y / 2., v_conn.x / 2.]);
                for i in 0..n {
                    let m = i as isize - n as isize / 2 + if n % 2 == 0 && i >= n / 2 { 1 } else { 0 };
                    if m == 0 {
                        if styles[i].1 {
                            self.draw_style(x2, None, None, x1, &styles[i].0);
                        } else {
                            self.draw_style(x1, None, None, x2, &styles[i].0);
                        }
                    } else {
                        let c = x1 + v_conn + m as f64 * v_perp;
                        if styles[i].1 {
                            self.draw_style(x2, Some(c), Some(c), x1, &styles[i].0);
                        } else {
                            self.draw_style(x1, Some(c), Some(c), x2, &styles[i].0);
                        }
                    }
                }
            }
        };
    }
    fn set_scaling(&mut self, positions: &NodeLayout);
}

pub(crate) struct SVGBackend {
    buf: String,
    scale: f64,
    shift: Vec2D,
}

impl SVGBackend {
    pub fn new() -> Self {
        Self {
            buf: String::new(),
            scale: 1.,
            shift: Vec2D::from([0., 0.]),
        }
    }

    pub(crate) fn draw_label_raw(&mut self, x: Vec2D, label: &str) {
        writeln!(
            self.buf,
            r#"<text x="{}" y="{}" font-size="15">{}</text>"#,
            x[0], x[1], label
        )
        .unwrap();
    }
}

impl DrawingBackend for SVGBackend {
    fn init(&mut self, n_rows: usize, n_cols: usize) {
        self.buf.clear();
        writeln!(
            self.buf,
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<!-- Automatically generated by FeynGraph -->
<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\">
<g>",
            n_cols * 300,
            n_rows * 325,
            n_cols * 300,
            n_rows * 325
        )
        .unwrap();
    }

    fn finish(&mut self) -> String {
        writeln!(
            self.buf,
            r#"</g>
</svg>
        "#
        )
        .unwrap();
        return std::mem::take(&mut self.buf);
    }

    fn init_group(&mut self, row: usize, col: usize) {
        writeln!(self.buf, "<g transform = \"translate({}, {})\">", col * 300, row * 325).unwrap();
    }

    fn finish_group(&mut self) {
        writeln!(self.buf, r#"</g>"#).unwrap();
    }

    fn draw_line(&mut self, x1: Vec2D, c1: Option<Vec2D>, c2: Option<Vec2D>, x2: Vec2D) {
        if let Some(c1) = c1
            && let Some(c2) = c2
        {
            writeln!(
                self.buf,
                r#"<path d="M {} {} C {} {}, {} {}, {} {}" stroke="black" stroke-width="2" fill="transparent"/>"#,
                self.scale * (x1[0] + self.shift.x),
                self.scale * (x1[1] + self.shift.y),
                self.scale * (c1[0] + self.shift.x),
                self.scale * (c1[1] + self.shift.y),
                self.scale * (c2[0] + self.shift.x),
                self.scale * (c2[1] + self.shift.y),
                self.scale * (x2[0] + self.shift.x),
                self.scale * (x2[1] + self.shift.y),
            )
            .unwrap();
        } else {
            writeln!(
                self.buf,
                r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="black" stroke-width="2"/>"#,
                self.scale * (x1[0] + self.shift.x),
                self.scale * (x1[1] + self.shift.y),
                self.scale * (x2[0] + self.shift.x),
                self.scale * (x2[1] + self.shift.y),
            )
            .unwrap();
        }
    }

    fn draw_self_loops(&mut self, x: Vec2D, r: f64, theta: f64, n_lines: usize) {
        let v = Vec2D::from_polar(r, theta);
        let v_perp = Vec2D::from([-v[1], v[0]]);
        for i in 1..=n_lines {
            self.draw_line(
                x,
                Some(x + i as f64 * (v + v_perp)),
                Some(x + i as f64 * (v - v_perp)),
                x,
            );
        }
    }

    fn draw_label(&mut self, x: Vec2D, label: &str) {
        writeln!(
            self.buf,
            r#"<text x="{}" y="{}" font-size="15">{}</text>"#,
            self.scale * (x[0] + self.shift.x),
            self.scale * (x[1] + self.shift.y),
            label
        )
        .unwrap();
    }

    fn draw_dashed(&mut self, x1: Vec2D, c1: Option<Vec2D>, c2: Option<Vec2D>, x2: Vec2D) {
        if let Some(c1) = c1
            && let Some(c2) = c2
        {
            writeln!(
                self.buf,
                r#"<path d="M {} {} C {} {}, {} {}, {} {}" stroke="black" stroke-width="2" fill="transparent" stroke-dasharray="5 3"/>"#,
                self.scale * (x1[0] + self.shift.x),
                self.scale * (x1[1] + self.shift.y),
                self.scale * (c1[0] + self.shift.x),
                self.scale * (c1[1] + self.shift.y),
                self.scale * (c2[0] + self.shift.x),
                self.scale * (c2[1] + self.shift.y),
                self.scale * (x2[0] + self.shift.x),
                self.scale * (x2[1] + self.shift.y),
            )
            .unwrap();
        } else {
            writeln!(
                self.buf,
                r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="black" stroke-width="2" stroke-dasharray="5 3"/>"#,
                self.scale * (x1[0] + self.shift.x),
                self.scale * (x1[1] + self.shift.y),
                self.scale * (x2[0] + self.shift.x),
                self.scale * (x2[1] + self.shift.y),
            )
            .unwrap();
        }
    }

    fn draw_dotted(&mut self, x1: Vec2D, c1: Option<Vec2D>, c2: Option<Vec2D>, x2: Vec2D) {
        if let Some(c1) = c1
            && let Some(c2) = c2
        {
            writeln!(
                self.buf,
                r#"<path d="M {} {} C {} {}, {} {}, {} {}" stroke="black" stroke-width="1.5" fill="transparent" stroke-dasharray="0 3" stroke-linecap="round"/>"#,
                self.scale * (x1[0] + self.shift.x),
                self.scale * (x1[1] + self.shift.y),
                self.scale * (c1[0] + self.shift.x),
                self.scale * (c1[1] + self.shift.y),
                self.scale * (c2[0] + self.shift.x),
                self.scale * (c2[1] + self.shift.y),
                self.scale * (x2[0] + self.shift.x),
                self.scale * (x2[1] + self.shift.y),
            )
            .unwrap();
        } else {
            writeln!(
                self.buf,
                r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="black" stroke-width="1.5" stroke-dasharray="0 3" stroke-linecap="round"/>"#,
                self.scale * (x1[0] + self.shift.x),
                self.scale * (x1[1] + self.shift.y),
                self.scale * (x2[0] + self.shift.x),
                self.scale * (x2[1] + self.shift.y),
            )
            .unwrap();
        }
    }

    fn draw_straight(&mut self, x1: Vec2D, c1: Option<Vec2D>, c2: Option<Vec2D>, x2: Vec2D) {
        let arrow_center;
        if let Some(c1) = c1
            && let Some(c2) = c2
        {
            arrow_center = cubic_bezier(x1, c1, c2, x2, 0.5);
            writeln!(
                self.buf,
                r#"<path d="M {} {} C {} {}, {} {}, {} {}" stroke="black" stroke-width="2" fill="transparent"/>"#,
                self.scale * (x1[0] + self.shift.x),
                self.scale * (x1[1] + self.shift.y),
                self.scale * (c1[0] + self.shift.x),
                self.scale * (c1[1] + self.shift.y),
                self.scale * (c2[0] + self.shift.x),
                self.scale * (c2[1] + self.shift.y),
                self.scale * (x2[0] + self.shift.x),
                self.scale * (x2[1] + self.shift.y),
            )
            .unwrap();
        } else {
            arrow_center = (x1 + x2) / 2.;
            writeln!(
                self.buf,
                r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="black" stroke-width="2"/>"#,
                self.scale * (x1[0] + self.shift.x),
                self.scale * (x1[1] + self.shift.y),
                self.scale * (x2[0] + self.shift.x),
                self.scale * (x2[1] + self.shift.y),
            )
            .unwrap();
        }
        let mut v = [x2[0] - x1[0], x2[1] - x1[1]];
        let v_norm = (v[0] * v[0] + v[1] * v[1]).sqrt();
        v = [v[0] / v_norm, v[1] / v_norm];
        let v_perp = [-v[1], v[0]];
        writeln!(
            self.buf,
            r#"<polygon points="{} {}, {} {}, {} {}" fill="black"/>"#,
            self.scale * (arrow_center[0] + self.shift.x) - 5. * (v[0] - v_perp[0]),
            self.scale * (arrow_center[1] + self.shift.y) - 5. * (v[1] - v_perp[1]),
            self.scale * (arrow_center[0] + self.shift.x) + 5. * v[0],
            self.scale * (arrow_center[1] + self.shift.y) + 5. * v[1],
            self.scale * (arrow_center[0] + self.shift.x) - 5. * (v[0] + v_perp[0]),
            self.scale * (arrow_center[1] + self.shift.y) - 5. * (v[1] + v_perp[1]),
        )
        .unwrap();
    }

    fn draw_wavy(&mut self, x1: Vec2D, c1: Option<Vec2D>, c2: Option<Vec2D>, x2: Vec2D) {
        write!(
            self.buf,
            r#"<path d="M {} {} "#,
            self.scale * (x1[0] + self.shift.x),
            self.scale * (x1[1] + self.shift.y)
        )
        .unwrap();
        let support_points = support_points(x1, c1, c2, x2);
        for i in 0..support_points.len() - 1 {
            let xi = support_points[i];
            let xf = support_points[i + 1];
            let v = [xf[0] - xi[0], xf[1] - xi[1]];
            let v_perp = [-v[1], v[0]];
            let c1 = [xi[0] + v[0] / 2. + v_perp[0], xi[1] + v[1] / 2. + v_perp[1]];
            let c2 = [xi[0] + v[0] / 2. - v_perp[0], xi[1] + v[1] / 2. - v_perp[1]];
            write!(
                self.buf,
                r#"C {} {}, {} {}, {} {} "#,
                self.scale * (c1[0] + self.shift.x),
                self.scale * (c1[1] + self.shift.y),
                self.scale * (c2[0] + self.shift.x),
                self.scale * (c2[1] + self.shift.y),
                self.scale * (xf[0] + self.shift.x),
                self.scale * (xf[1] + self.shift.y),
            )
            .unwrap();
        }
        writeln!(self.buf, r#"" stroke="black" stroke-width="2" fill="transparent"/>"#).unwrap();
    }

    fn draw_curly(&mut self, x1: Vec2D, c1: Option<Vec2D>, c2: Option<Vec2D>, x2: Vec2D) {
        write!(
            self.buf,
            r#"<path d="M {} {} "#,
            self.scale * (x1[0] + self.shift.x),
            self.scale * (x1[1] + self.shift.y)
        )
        .unwrap();
        let support_points = support_points(
            self.scale * (x1 + self.shift),
            c1.map(|c1| self.scale * (c1 + self.shift)),
            c2.map(|c2| self.scale * (c2 + self.shift)),
            self.scale * (x2 + self.shift),
        );
        for i in 0..support_points.len() - 1 {
            let xi = support_points[i];
            let xf = support_points[i + 1];
            let d = (xf - xi).norm();
            let theta = (xf[1] - xi[1]).atan2(xf[0] - xi[0]);
            let s1 = xi + d * CURL_SUPPORT[0].rotate(theta);
            let c11 = xi + CURL_AMPLITUDE * CURL_CONTROL[0].rotate(theta);
            let c12 = s1 + CURL_AMPLITUDE * CURL_CONTROL[1].rotate(theta);
            if i + 2 == support_points.len() {
                write!(
                    self.buf,
                    r#"C {} {}, {} {}, {} {}"#,
                    c11[0], c11[1], c12[0], c12[1], xf[0], xf[1]
                )
                .unwrap();
            } else {
                let s2 = xi + d * CURL_SUPPORT[1].rotate(theta);
                let c21 = s1 + CURL_AMPLITUDE * CURL_CONTROL[2].rotate(theta);
                let c22 = s2 + CURL_AMPLITUDE * CURL_CONTROL[3].rotate(theta);
                write!(
                    self.buf,
                    r#"C {} {}, {} {}, {} {} C {} {}, {} {}, {} {} "#,
                    c11[0], c11[1], c12[0], c12[1], s1[0], s1[1], c21[0], c21[1], c22[0], c22[1], s2[0], s2[1]
                )
                .unwrap();
            }
        }
        writeln!(self.buf, r#"" stroke="black" stroke-width="2" fill="transparent"/>"#).unwrap();
    }

    fn draw_scurly(&mut self, x1: Vec2D, c1: Option<Vec2D>, c2: Option<Vec2D>, x2: Vec2D) {
        self.draw_curly(x1, c1, c2, x2);
        self.draw_line(x1, c1, c2, x2);
    }

    fn draw_swavy(&mut self, x1: Vec2D, c1: Option<Vec2D>, c2: Option<Vec2D>, x2: Vec2D) {
        self.draw_wavy(x1, c1, c2, x2);
        self.draw_line(x1, c1, c2, x2);
    }

    fn draw_double(&mut self, x1: Vec2D, c1: Option<Vec2D>, c2: Option<Vec2D>, x2: Vec2D) {
        let v = (x2 - x1).normalize();
        let v_perp = 0.25 * Vec2D::from([-v[1], v[0]]);
        self.draw_line(
            x1 + v_perp,
            c1.map(|c1| c1 + v_perp),
            c2.map(|c2| c2 + v_perp),
            x2 + v_perp,
        );
        self.draw_line(
            x1 - v_perp,
            c1.map(|c1| c1 - v_perp),
            c2.map(|c2| c2 - v_perp),
            x2 - v_perp,
        );
    }

    fn set_scaling(&mut self, positions: &NodeLayout) {
        let xmin = positions
            .iter()
            .map(|v| v.x)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let xmax = positions
            .iter()
            .map(|v| v.x)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let ymin = positions
            .iter()
            .map(|v| v.y)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let ymax = positions
            .iter()
            .map(|v| v.y)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let x_scale = 0.8 * 300. / if xmax == xmin { 3. } else { xmax - xmin };
        let y_scale = 0.8 * 300. / if xmax == xmin { 3. } else { ymax - ymin };
        let scale = x_scale.min(y_scale);
        self.scale = scale;
        self.shift.x = -(xmax + xmin) / 2. + 150. / scale;
        self.shift.y = -(ymax + ymin) / 2. + 175. / scale;
    }
}

pub(crate) struct TikzBackend {
    nodes: Vec<Vec2D>,
    node_buffer: String,
    edge_buffer: String,
    scale: f64,
    shift: Vec2D,
}

impl TikzBackend {
    pub(crate) fn new() -> Self {
        Self {
            nodes: Vec::new(),
            node_buffer: String::new(),
            edge_buffer: String::new(),
            scale: 1.0,
            shift: Vec2D::from([0., 0.]),
        }
    }

    fn push_node(&mut self, x: Vec2D, label: Option<&str>) -> usize {
        match self.nodes.iter().position(|&r| r == x) {
            Some(i) => i,
            None => {
                self.nodes.push(x);
                let i = self.nodes.len() - 1;
                writeln!(
                    self.node_buffer,
                    "        \\node [style = none] ({}) at ({:.4}, {:.4}) {{{}}};",
                    i,
                    self.scale * (x[0] + self.shift.x),
                    self.scale * (x[1] + self.shift.y),
                    label.unwrap_or("")
                )
                .unwrap();
                return i;
            }
        }
    }
}

impl DrawingBackend for TikzBackend {
    fn init(&mut self, _n_rows: usize, _n_cols: usize) {
        return;
    }

    fn finish(&mut self) -> String {
        let mut res = String::new();
        writeln!(
            res,
            r"\begin{{tikzpicture}}
    \begin{{pgfonlayer}}{{nodelayer}}
{}
    \end{{pgfonlayer}}
    \begin{{pgfonlayer}}{{edgelayer}}
{}
    \end{{pgfonlayer}}
\end{{tikzpicture}}
        ",
            self.node_buffer, self.edge_buffer
        )
        .unwrap();
        self.nodes.clear();
        self.node_buffer.clear();
        self.edge_buffer.clear();
        return res;
    }

    fn init_group(&mut self, row: usize, col: usize) {
        todo!();
    }

    fn finish_group(&mut self) {
        todo!();
    }

    fn draw_line(&mut self, x1: Vec2D, c1: Option<Vec2D>, c2: Option<Vec2D>, x2: Vec2D) {
        let i = self.push_node(x1, None);
        let j = self.push_node(x2, None);
        if let Some(c1) = c1
            && let Some(c2) = c2
        {
            writeln!(
                self.edge_buffer,
                "      \\draw ({}.center) .. controls ({:.4}, {:.4}) and ({:.4}, {:.4}) .. ({}.center);",
                i, c1[0], c1[1], c2[0], c2[1], j
            )
            .unwrap();
        } else {
            writeln!(self.edge_buffer, "        \\draw ({}.center) to ({}.center);", i, j).unwrap();
        }
    }

    fn draw_label(&mut self, x: Vec2D, label: &str) {
        self.push_node(x, Some(label));
    }

    fn draw_self_loops(&mut self, x: Vec2D, r: f64, theta: f64, n_lines: usize) {
        todo!()
    }

    fn draw_dashed(&mut self, x1: Vec2D, c1: Option<Vec2D>, c2: Option<Vec2D>, x2: Vec2D) {
        let i = self.push_node(x1, None);
        let j = self.push_node(x2, None);
        if let Some(c1) = c1
            && let Some(c2) = c2
        {
            writeln!(
                self.edge_buffer,
                "      \\draw[style=dashed] ({}.center) .. controls ({:.4}, {:.4}) and ({:.4}, {:.4}) .. ({}.center);",
                i, c1[0], c1[1], c2[0], c2[1], j
            )
            .unwrap();
        } else {
            writeln!(
                self.edge_buffer,
                "        \\draw[style=dashed] ({}.center) to ({}.center);",
                i, j
            )
            .unwrap();
        }
    }

    fn draw_dotted(&mut self, x1: Vec2D, c1: Option<Vec2D>, c2: Option<Vec2D>, x2: Vec2D) {
        let i = self.push_node(x1, None);
        let j = self.push_node(x2, None);
        if let Some(c1) = c1
            && let Some(c2) = c2
        {
            writeln!(
                self.edge_buffer,
                "      \\draw[style=dotted] ({}.center) .. controls ({:.4}, {:.4}) and ({:.4}, {:.4}) .. ({}.center);",
                i, c1[0], c1[1], c2[0], c2[1], j
            )
            .unwrap();
        } else {
            writeln!(
                self.edge_buffer,
                "        \\draw[style=dotted] ({}.center) to ({}.center);",
                i, j
            )
            .unwrap();
        }
    }

    fn draw_straight(&mut self, x1: Vec2D, c1: Option<Vec2D>, c2: Option<Vec2D>, x2: Vec2D) {
        let i = self.push_node(x1, None);
        let j = self.push_node(x2, None);
        if let Some(c1) = c1
            && let Some(c2) = c2
        {
            writeln!(
                self.edge_buffer,
                "      \\draw[style=straight] ({}.center) .. controls ({:.4}, {:.4}) and ({:.4}, {:.4}) .. ({}.center);",
                i, c1[0], c1[1], c2[0], c2[1], j
            )
            .unwrap();
        } else {
            writeln!(
                self.edge_buffer,
                "        \\draw[style=straight] ({}.center) to ({}.center);",
                i, j
            )
            .unwrap();
        }
    }

    fn draw_wavy(&mut self, x1: Vec2D, c1: Option<Vec2D>, c2: Option<Vec2D>, x2: Vec2D) {
        let i = self.push_node(x1, None);
        let j = self.push_node(x2, None);
        if let Some(c1) = c1
            && let Some(c2) = c2
        {
            writeln!(
                self.edge_buffer,
                "      \\draw[style=wavy] ({}.center) .. controls ({:.4}, {:.4}) and ({:.4}, {:.4}) .. ({}.center);",
                i, c1[0], c1[1], c2[0], c2[1], j
            )
            .unwrap();
        } else {
            writeln!(
                self.edge_buffer,
                "        \\draw[style=wavy] ({}.center) to ({}.center);",
                i, j
            )
            .unwrap();
        }
    }

    fn draw_curly(&mut self, x1: Vec2D, c1: Option<Vec2D>, c2: Option<Vec2D>, x2: Vec2D) {
        let i = self.push_node(x1, None);
        let j = self.push_node(x2, None);
        if let Some(c1) = c1
            && let Some(c2) = c2
        {
            writeln!(
                self.edge_buffer,
                "      \\draw[style=curly] ({}.center) .. controls ({:.4}, {:.4}) and ({:.4}, {:.4}) .. ({}.center);",
                i, c1[0], c1[1], c2[0], c2[1], j
            )
            .unwrap();
        } else {
            writeln!(
                self.edge_buffer,
                "        \\draw[style=curly] ({}.center) to ({}.center);",
                i, j
            )
            .unwrap();
        }
    }

    fn draw_scurly(&mut self, x1: Vec2D, c1: Option<Vec2D>, c2: Option<Vec2D>, x2: Vec2D) {
        let i = self.push_node(x1, None);
        let j = self.push_node(x2, None);
        if let Some(c1) = c1
            && let Some(c2) = c2
        {
            writeln!(
                self.edge_buffer,
                "      \\draw[style=scurly] ({}.center) .. controls ({:.4}, {:.4}) and ({:.4}, {:.4}) .. ({}.center);",
                i, c1[0], c1[1], c2[0], c2[1], j
            )
            .unwrap();
        } else {
            writeln!(
                self.edge_buffer,
                "        \\draw[style=scurly] ({}.center) to ({}.center);",
                i, j
            )
            .unwrap();
        }
    }

    fn draw_swavy(&mut self, x1: Vec2D, c1: Option<Vec2D>, c2: Option<Vec2D>, x2: Vec2D) {
        let i = self.push_node(x1, None);
        let j = self.push_node(x2, None);
        if let Some(c1) = c1
            && let Some(c2) = c2
        {
            writeln!(
                self.edge_buffer,
                "      \\draw[style=swavy] ({}.center) .. controls ({:.4}, {:.4}) and ({:.4}, {:.4}) .. ({}.center);",
                i, c1[0], c1[1], c2[0], c2[1], j
            )
            .unwrap();
        } else {
            writeln!(
                self.edge_buffer,
                "        \\draw[style=swavy] ({}.center) to ({}.center);",
                i, j
            )
            .unwrap();
        }
    }

    fn draw_double(&mut self, x1: Vec2D, c1: Option<Vec2D>, c2: Option<Vec2D>, x2: Vec2D) {
        let i = self.push_node(x1, None);
        let j = self.push_node(x2, None);
        if let Some(c1) = c1
            && let Some(c2) = c2
        {
            writeln!(
                self.edge_buffer,
                "      \\draw[style=double] ({}.center) .. controls ({:.4}, {:.4}) and ({:.4}, {:.4}) .. ({}.center);",
                i, c1[0], c1[1], c2[0], c2[1], j
            )
            .unwrap();
        } else {
            writeln!(
                self.edge_buffer,
                "        \\draw[style=double] ({}.center) to ({}.center);",
                i, j
            )
            .unwrap();
        }
    }

    fn set_scaling(&mut self, positions: &NodeLayout) {
        let xmin = positions
            .iter()
            .map(|v| v.x)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let xmax = positions
            .iter()
            .map(|v| v.x)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let ymin = positions
            .iter()
            .map(|v| v.y)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let ymax = positions
            .iter()
            .map(|v| v.y)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let x_scale = 10. / (xmax - xmin);
        let y_scale = 10. / (ymax - ymin);
        let scale = x_scale.min(y_scale);
        self.scale = scale;
        self.shift.x = -(xmax + xmin) / 2. + 5. / scale;
        self.shift.y = -(ymax + ymin) / 2. + 5. / scale;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tikz_test() {
        let mut tikz = TikzBackend::new();
        tikz.init(1, 1);
        tikz.draw_line([0., 0.].into(), None, None, [1., -1.].into());
        tikz.draw_line([0., -2.].into(), None, None, [1., -1.].into());
        tikz.draw_line([1., -1.].into(), None, None, [2., -1.].into());
        tikz.draw_line([2., -1.].into(), None, None, [3., 0.].into());
        tikz.draw_line([2., -1.].into(), None, None, [3., -2.].into());
        let res = tikz.finish();
        println!("{}", res);
    }
}
