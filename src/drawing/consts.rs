use super::math::Vec2D;
use std::f64::consts::PI;

// pub(crate) const N_SUPPORT: usize = 16;
// pub(crate) const SCAN_MULTIPLIER: usize = 10;

// pub(crate) const TITLE_SIZE: f64 = 0.15;
// pub(crate) const BORDER_SIZE: f64 = 0.05;

// pub(crate) const ARROW_SIZE: f64 = 0.015;
// pub(crate) const WAVY_AMPLITUDE: f64 = 0.075;
// pub(crate) const CURLY_AMPLITUDE: f64 = 0.065;
// pub(crate) const SELF_LOOP_SIZE: f64 = 0.3;

pub(crate) const ARCLEN_RESOLUTION: usize = 100;

/// Support and control points for a cubic Bezier approximation of the parametric curve
/// $$\begin{pmatrix} \left(2 + \frac{3}{4} t - 2 \cos t\right) / \left(2 + \frac{9 \pi}{4}\right) \\ -\frac{2}{2} \sin t\end{pmatrix}$$
pub(crate) const CURL_SUPPORT: [Vec2D; 2] = [
    Vec2D {
        x: (32. + 6. * PI) / (16. + 9. * PI),
        y: 0.,
    },
    Vec2D {
        x: 12. * PI / (16. + 9. * PI),
        y: 0.,
    },
];
pub(crate) const CURL_CONTROL: [Vec2D; 4] = [
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
