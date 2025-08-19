use crate::drawing::N_SUPPORT;
use itertools::Itertools;
use std::{
    iter::Sum,
    ops::{Add, Deref, Div, Index, IndexMut, Mul, Sub},
};

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Matrix {
    pub(crate) dim: (usize, usize),
    pub(crate) data: Vec<Vec<f64>>,
}

impl Matrix {
    pub(crate) fn dim(&self) -> (usize, usize) {
        return self.dim;
    }

    pub(crate) fn submatrix(&self, min: usize, max: usize) -> Matrix {
        debug_assert_eq!(self.dim.0, self.dim.1);
        Matrix {
            dim: (max - min, max - min),
            data: self.data[min..max].iter().map(|row| row[min..max].to_vec()).collect(),
        }
    }

    pub(crate) fn take_rows(&self, min: usize, max: usize) -> Matrix {
        Matrix {
            dim: (max - min, self.dim.1),
            data: self.data[min..max].to_owned(),
        }
    }

    pub(crate) fn map<F: Fn(&f64) -> f64>(&self, f: F) -> Matrix {
        return Matrix {
            dim: self.dim,
            data: self
                .data
                .iter()
                .map(|row| row.iter().map(&f).collect_vec())
                .collect_vec(),
        };
    }
}

impl Deref for Matrix {
    type Target = Vec<Vec<f64>>;

    fn deref(&self) -> &Self::Target {
        return &self.data;
    }
}

#[derive(Clone)]
pub(crate) struct Vector {
    pub(crate) inner: Vec<f64>,
}

impl Vector {
    pub(crate) fn sq_norm(&self) -> f64 {
        return self.inner.iter().map(|x| x * x).sum();
    }
}

impl Deref for Vector {
    type Target = Vec<f64>;

    fn deref(&self) -> &Self::Target {
        return &self.inner;
    }
}

impl Add for Vector {
    type Output = Vector;

    fn add(self, rhs: Self) -> Self::Output {
        return Vector {
            inner: self
                .inner
                .into_iter()
                .zip(rhs.inner.iter())
                .map(|(x, y)| x + y)
                .collect(),
        };
    }
}

impl Sub for Vector {
    type Output = Vector;

    fn sub(self, rhs: Self) -> Self::Output {
        return Vector {
            inner: self
                .inner
                .into_iter()
                .zip(rhs.inner.iter())
                .map(|(x, y)| x - y)
                .collect(),
        };
    }
}

impl Mul<Vector> for f64 {
    type Output = Vector;

    fn mul(self, rhs: Vector) -> Self::Output {
        return Vector {
            inner: rhs.inner.into_iter().map(|x| self * x).collect(),
        };
    }
}

impl Mul<Vector> for Vector {
    type Output = f64;

    fn mul(self, rhs: Vector) -> Self::Output {
        return rhs.inner.into_iter().zip(self.inner.iter()).map(|(x, y)| x * y).sum();
    }
}

impl Mul<Vector> for &Matrix {
    type Output = Vector;

    fn mul(self, rhs: Vector) -> Self::Output {
        debug_assert_eq!(self.dim.1, rhs.len());
        Vector {
            inner: self
                .data
                .iter()
                .map(|row| row.iter().zip(rhs.iter()).map(|(a, x)| *a * x).sum::<f64>())
                .collect_vec(),
        }
    }
}

impl Add for &Vector {
    type Output = Vector;

    fn add(self, rhs: Self) -> Self::Output {
        return Vector {
            inner: self.inner.iter().zip(rhs.inner.iter()).map(|(x, y)| x + y).collect(),
        };
    }
}

impl Sub for &Vector {
    type Output = Vector;

    fn sub(self, rhs: Self) -> Self::Output {
        return Vector {
            inner: self.inner.iter().zip(rhs.inner.iter()).map(|(x, y)| x - y).collect(),
        };
    }
}

impl Mul<&Vector> for f64 {
    type Output = Vector;

    fn mul(self, rhs: &Vector) -> Self::Output {
        return Vector {
            inner: rhs.inner.iter().map(|x| self * x).collect(),
        };
    }
}

impl Mul<&Vector> for &Vector {
    type Output = f64;

    fn mul(self, rhs: &Vector) -> Self::Output {
        return rhs.inner.iter().zip(self.inner.iter()).map(|(x, y)| x * y).sum();
    }
}

impl Mul<&Vector> for &Matrix {
    type Output = Vector;

    fn mul(self, rhs: &Vector) -> Self::Output {
        debug_assert_eq!(self.dim.1, rhs.len());
        Vector {
            inner: self
                .data
                .iter()
                .map(|row| row.iter().zip(rhs.iter()).map(|(a, x)| *a * *x).sum::<f64>())
                .collect_vec(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Vec2D {
    pub(crate) x: f64,
    pub(crate) y: f64,
}

impl From<[f64; 2]> for Vec2D {
    fn from(value: [f64; 2]) -> Self {
        return Vec2D {
            x: value[0],
            y: value[1],
        };
    }
}

impl Vec2D {
    pub(crate) fn sq_norm(&self) -> f64 {
        return self.x * self.x + self.y * self.y;
    }
    pub(crate) fn norm(&self) -> f64 {
        return self.x.hypot(self.y);
    }
    pub(crate) fn normalize(self) -> Self {
        let norm = self.norm();
        return Self {
            x: self.x / norm,
            y: self.y / norm,
        };
    }
    pub(crate) fn from_polar(r: f64, theta: f64) -> Self {
        let (sin, cos) = theta.sin_cos();
        Vec2D { x: r * cos, y: r * sin }
    }
    pub(crate) fn rotate(self, theta: f64) -> Self {
        let (sin, cos) = theta.sin_cos();
        return Self::from([self.x * cos - self.y * sin, self.x * sin + self.y * cos]);
    }
}

impl Index<usize> for Vec2D {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            _ => panic!(
                "Index out of bounds: trying to access index {}, but Vec2D only has two entries",
                index
            ),
        }
    }
}

impl IndexMut<usize> for Vec2D {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            _ => panic!(
                "Index out of bounds: trying to access index {}, but Vec2D only has two entries",
                index
            ),
        }
    }
}

impl Add<Vec2D> for Vec2D {
    type Output = Vec2D;

    fn add(self, rhs: Vec2D) -> Self::Output {
        Vec2D {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl Sub<Vec2D> for Vec2D {
    type Output = Vec2D;

    fn sub(self, rhs: Vec2D) -> Self::Output {
        Vec2D {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

impl Mul<Vec2D> for Vec2D {
    type Output = f64;

    fn mul(self, rhs: Vec2D) -> Self::Output {
        return self.x * rhs.x + self.y * rhs.y;
    }
}

impl Mul<f64> for Vec2D {
    type Output = Vec2D;

    fn mul(self, rhs: f64) -> Self::Output {
        Vec2D {
            x: rhs * self.x,
            y: rhs * self.y,
        }
    }
}

impl Mul<Vec2D> for f64 {
    type Output = Vec2D;

    fn mul(self, rhs: Vec2D) -> Self::Output {
        Vec2D {
            x: self * rhs.x,
            y: self * rhs.y,
        }
    }
}

impl Div<f64> for Vec2D {
    type Output = Vec2D;

    fn div(self, rhs: f64) -> Self::Output {
        Vec2D {
            x: self.x / rhs,
            y: self.y / rhs,
        }
    }
}

impl<'a> Sum<&'a Vec2D> for Vec2D {
    fn sum<I: Iterator<Item = &'a Self>>(mut iter: I) -> Self {
        let mut res;
        if let Some(v) = iter.next() {
            res = *v;
        } else {
            return Vec2D { x: 0., y: 0. };
        }
        for v in iter {
            res = res + *v;
        }
        return res;
    }
}

pub(crate) fn cubic_bezier(x1: Vec2D, c1: Vec2D, c2: Vec2D, x2: Vec2D, t: f64) -> Vec2D {
    let tp = 1. - t;
    return Vec2D::from([
        tp * (tp * (tp * x1[0] + t * c1[0]) + t * (tp * c1[0] + t * c2[0]))
            + t * (tp * (tp * c1[0] + t * c2[0]) + t * (tp * c2[0] + t * x2[0])),
        tp * (tp * (tp * x1[1] + t * c1[1]) + t * (tp * c1[1] + t * c2[1]))
            + t * (tp * (tp * c1[1] + t * c2[1]) + t * (tp * c2[1] + t * x2[1])),
    ]);
}

pub(crate) fn support_points(x1: Vec2D, c1: Option<Vec2D>, c2: Option<Vec2D>, x2: Vec2D) -> Vec<Vec2D> {
    return if let Some(c1) = c1
        && let Some(c2) = c2
    {
        (0..=N_SUPPORT)
            .map(|i| cubic_bezier(x1, c1, c2, x2, i as f64 / N_SUPPORT as f64))
            .collect_vec()
    } else {
        let v = x2 - x1;
        (0..=N_SUPPORT)
            .map(|i| x1 + (i as f64 / N_SUPPORT as f64) * v)
            .collect_vec()
    };
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use crate::drawing::math::cubic_bezier;

    #[test]
    fn bezier_test() {
        let x1 = [1., 1.].into();
        let c1 = [3., -2.].into();
        let c2 = [-2., 4.].into();
        let x2 = [5., 2.].into();
        let ref_val = [1.393152922296, 0.3259132884720002];
        let val = cubic_bezier(x1, c1, c2, x2, 0.3194);
        assert_relative_eq!(ref_val[0], val[0]);
        assert_relative_eq!(ref_val[1], val[1]);
    }
}
