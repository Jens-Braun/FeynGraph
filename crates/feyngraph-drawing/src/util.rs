use itertools::Itertools;
use std::f64;

use super::{components::Segment, math::Vec2D, style::Theme};

pub(crate) fn multi_path(x1: Vec2D, x2: Vec2D, n: usize) -> Vec<Segment> {
    let mut paths = Vec::with_capacity(n);
    let v_conn = (x2 - x1) / 2.;
    for i in 0..n {
        let m = i as isize - n as isize / 2 + if n % 2 == 0 && i >= n / 2 { 1 } else { 0 };
        if m == 0 {
            paths.push(Segment::Line(x2));
        } else {
            let c = x1 + v_conn + 1.5 * m as f64 * v_conn.perp();
            paths.push(Segment::QuadraticBezier(c, x2));
        }
    }
    return paths;
}

pub(crate) fn self_paths(x: Vec2D, neighbors: &[Vec2D], scale: f64, n: usize) -> Vec<Segment> {
    let angles = neighbors
        .iter()
        .map(|v| (*v - x).arg())
        .sorted_by(f64::total_cmp)
        .collect::<Vec<_>>();
    let mut max_diff = angles[0] + 2. * f64::consts::PI - angles[angles.len() - 1];
    let mut angle = angles[angles.len() - 1] + 0.5 * max_diff;
    if angle > 2. * f64::consts::PI {
        angle -= 2. * f64::consts::PI
    };
    for (a, b) in angles.into_iter().tuple_windows() {
        let diff = (b - a).abs();
        if diff > max_diff {
            max_diff = diff;
            angle = a + 0.5 * diff;
        }
    }
    let mut segments = Vec::with_capacity(n);
    let v = Vec2D::from_polar(scale * Theme::get_global().self_loop_size, angle);
    let v1 = v + v.perp();
    let v2 = v - v.perp();
    for i in 1..=n {
        segments.push(Segment::CubicBezier(x + i as f64 * v1, x + i as f64 * v2, x))
    }
    return segments;
}
