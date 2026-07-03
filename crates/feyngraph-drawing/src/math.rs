use itertools::Itertools;
use std::{
    f64,
    iter::Sum,
    ops::{Add, Deref, Div, Index, IndexMut, Mul, Neg, Sub},
};

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Matrix {
    pub(crate) dim: (usize, usize),
    pub(crate) data: Vec<Vec<f64>>,
}

impl Matrix {
    pub(crate) fn submatrix(&self, min: usize, max: usize) -> Matrix {
        debug_assert_eq!(self.dim.0, self.dim.1);
        Matrix {
            dim: (max - min, max - min),
            data: self.data[min..max].iter().map(|row| row[min..max].to_vec()).collect(),
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
pub struct Vec2D {
    pub x: f64,
    pub y: f64,
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
    pub(crate) fn scale(&mut self, scale_x: f64, scale_y: f64) {
        self.x *= scale_x;
        self.y *= scale_y;
    }
    pub(crate) fn perp(&self) -> Self {
        Self { x: -self.y, y: self.x }
    }
    pub(crate) fn arg(&self) -> f64 {
        let tmp = self.y.atan2(self.x);
        if tmp < 0. { tmp + f64::consts::PI * 2. } else { tmp }
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

impl Neg for Vec2D {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self { x: -self.x, y: -self.y }
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
