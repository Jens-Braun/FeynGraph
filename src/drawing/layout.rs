#![allow(non_snake_case)]

use itertools::Itertools;
use std::ops::Deref;

use super::math::{Matrix, Vec2D, Vector};
use crate::diagram::Diagram;
use crate::diagram::view::DiagramView;
use crate::topology::Topology;

pub(crate) struct TopologyLayout<'a> {
    topo: &'a Topology,
    distances: Option<Vec<Vec<usize>>>,
}

impl TopologyLayout<'_> {
    fn bfs(&self, start: usize) -> Vec<usize> {
        let dim = self.topo.nodes.len();
        let mut d = vec![0; dim];
        let mut distance: usize = 1;
        let mut seen = vec![false; dim];
        seen[start] = true;
        let mut to_visit = Some(vec![start]);
        while let Some(ref nodes) = to_visit {
            let mut next = Vec::new();
            for n in nodes {
                for adj in self.topo.nodes[*n].adjacent_nodes.iter() {
                    if seen[*adj] {
                        continue;
                    }
                    seen[*adj] = true;
                    d[*adj] = distance;
                    next.push(*adj);
                }
            }
            distance += 1;
            if next.is_empty() {
                to_visit = None;
            } else {
                to_visit = Some(next);
            }
        }
        return d;
    }

    fn calculate_distances(&mut self) {
        let dim = self.topo.nodes.len();
        let mut d = Vec::with_capacity(dim);
        for i in 0..dim {
            d.push(self.bfs(i));
        }
        self.distances = Some(d);
    }
}

impl Layout for TopologyLayout<'_> {
    fn distance_matrix(&mut self) -> Matrix {
        if self.distances.is_none() {
            self.calculate_distances();
        }
        return Matrix {
            dim: (self.topo.nodes.len(), self.topo.nodes.len()),
            data: self
                .distances
                .as_ref()
                .unwrap()
                .iter()
                .map(|row| row.iter().map(|x| *x as f64).collect_vec())
                .collect_vec(),
        };
    }

    /// The nodes are initially placed on a grid, where the external nodes $i$ are fixed to $x_i = 0$ and all
    /// other nodes $j$ are placed at $x_j = \min_{i} d(i, j)$ with the graph theoretical distance $d$.
    fn initial_positions(&mut self) -> Vec<Vec2D> {
        if self.distances.is_none() {
            self.calculate_distances();
        }
        let d = self.distances.as_ref().unwrap();
        let n_in = self.topo.n_external / 2;
        let class_distances = d
            .iter()
            .map(|row| *row.iter().take(n_in).min().unwrap_or(&0))
            .collect_vec();
        let mut class_members = vec![Vec::new(); *class_distances.iter().max().unwrap() + 1];
        for (i, d) in class_distances.iter().enumerate() {
            class_members[*d].push(i);
        }
        let mut vecs = vec![Vec2D { x: 0., y: 0. }; class_distances.len()];
        let zero_shift = class_members[0].len() as f64 / 2.;
        for (i, members) in class_members.into_iter().enumerate() {
            let shift = members.len() as f64 / 2.;
            for (j, k) in members.into_iter().enumerate() {
                vecs[k] = Vec2D {
                    x: i as f64,
                    y: j as f64 - shift + zero_shift,
                }
            }
        }
        return vecs;
    }

    fn n_in(&self) -> usize {
        return self.topo.n_external / 2;
    }

    fn n_out(&self) -> usize {
        return self.topo.n_external - self.n_in();
    }
}

impl<'a> From<&'a Topology> for TopologyLayout<'a> {
    fn from(topo: &'a Topology) -> Self {
        Self { topo, distances: None }
    }
}

pub(crate) struct DiagramLayout<'a> {
    diag: &'a Diagram,
    distances: Option<Vec<Vec<usize>>>,
}

impl DiagramLayout<'_> {
    fn bfs(&self, start: isize) -> Vec<usize> {
        let dim = self.diag.n_ext() + self.diag.vertices.len();
        let n_ext = self.diag.n_ext();
        let mut d = vec![0; dim];
        let mut distance: usize = 1;
        let mut seen = vec![false; dim];
        seen[(start + n_ext as isize) as usize] = true;
        let mut to_visit;
        if start < 0 {
            let start = (start + n_ext as isize) as usize;
            distance += 1;
            if start >= self.diag.incoming_legs.len() {
                to_visit = Some(vec![
                    self.diag.outgoing_legs[start - self.diag.incoming_legs.len()].vertex,
                ])
            } else {
                to_visit = Some(vec![self.diag.incoming_legs[start].vertex])
            }
            seen[to_visit.as_ref().unwrap()[0] + n_ext] = true;
            d[to_visit.as_ref().unwrap()[0] + n_ext] = 1;
        } else {
            to_visit = Some(vec![start as usize])
        };
        seen[(start + n_ext as isize) as usize] = true;
        while let Some(ref vertices) = to_visit {
            let mut next = Vec::new();
            for v in vertices {
                for prop in self.diag.vertices[*v].propagators.iter() {
                    if *prop < 0 {
                        if seen[(*prop + n_ext as isize) as usize] {
                            continue;
                        }
                        seen[(*prop + n_ext as isize) as usize] = true;
                        d[(*prop + n_ext as isize) as usize] = distance;
                    } else {
                        let adj = if self.diag.propagators[*prop as usize].vertices[0] == *v {
                            self.diag.propagators[*prop as usize].vertices[1]
                        } else {
                            self.diag.propagators[*prop as usize].vertices[0]
                        };
                        if seen[adj + n_ext] {
                            continue;
                        }
                        seen[adj + n_ext] = true;
                        d[adj + n_ext] = distance;
                        next.push(adj);
                    }
                }
            }
            distance += 1;
            if next.is_empty() {
                to_visit = None;
            } else {
                to_visit = Some(next);
            }
        }
        return d;
    }

    fn calculate_distances(&mut self) {
        let dim = self.diag.n_ext() + self.diag.vertices.len();
        let mut d = Vec::with_capacity(dim);
        for i in (-(self.diag.n_ext() as isize))..(self.diag.vertices.len() as isize) {
            d.push(self.bfs(i));
        }
        self.distances = Some(d);
    }
}

impl Layout for DiagramLayout<'_> {
    fn distance_matrix(&mut self) -> Matrix {
        if self.distances.is_none() {
            self.calculate_distances();
        }
        return Matrix {
            dim: (
                self.diag.n_ext() + self.diag.vertices.len(),
                self.diag.n_ext() + self.diag.vertices.len(),
            ),
            data: self
                .distances
                .as_ref()
                .unwrap()
                .iter()
                .map(|row| row.iter().map(|x| *x as f64).collect_vec())
                .collect_vec(),
        };
    }

    /// The vertices are initially placed on a grid, where the incoming nodes $i$ are fixed to $x_i = 0$ and all
    /// other nodes $j$ are placed at $x_j = \min_{i} d(i, j)$ with the graph theoretical distance $d$.
    fn initial_positions(&mut self) -> Vec<Vec2D> {
        if self.distances.is_none() {
            self.calculate_distances();
        }
        let d = self.distances.as_ref().unwrap();
        let class_distances = d
            .iter()
            .map(|row| *row.iter().take(self.diag.n_in()).min().unwrap_or(&0))
            .collect_vec();
        let mut class_members = vec![Vec::new(); *class_distances.iter().max().unwrap() + 1];
        for (i, d) in class_distances.iter().enumerate() {
            class_members[*d].push(i);
        }
        let mut vecs = vec![Vec2D { x: 0., y: 0. }; class_distances.len()];
        let zero_shift = class_members[0].len() as f64 / 2.;
        for (i, members) in class_members.into_iter().enumerate() {
            let shift = members.len() as f64 / 2.;
            for (j, k) in members.into_iter().enumerate() {
                vecs[k] = Vec2D {
                    x: i as f64,
                    y: j as f64 - shift + zero_shift,
                }
            }
        }
        return vecs;
    }

    fn n_in(&self) -> usize {
        return self.diag.n_in();
    }

    fn n_out(&self) -> usize {
        return self.diag.n_out();
    }
}

impl<'a> From<&'a DiagramView<'a>> for DiagramLayout<'a> {
    fn from(view: &'a DiagramView) -> Self {
        return Self {
            diag: view.diagram,
            distances: None,
        };
    }
}

/// Automatic layouting using stress majorization (10.1007/978-3-540-31843-9_25). The variable naming in this module
/// follows the conventions of the given paper (except $\delta \rightarrow \alpha$).
pub(crate) trait Layout {
    /// The distance matrix $d_{ij}$
    fn distance_matrix(&mut self) -> Matrix;
    /// Initial condition for the layout
    fn initial_positions(&mut self) -> Vec<Vec2D>;
    /// Number of incoming nodes
    fn n_in(&self) -> usize;
    /// Number of outgoing nodes
    fn n_out(&self) -> usize;

    fn layout(&mut self) -> NodeLayout {
        let n_ext = self.n_in() + self.n_out();
        let d = self.distance_matrix();
        let w = d.map(|x| if *x != 0. { 1. / ((*x) * (*x)) } else { 0. });
        let alpha = d.map(|x| if *x != 0. { 1. / *x } else { 0. });
        let mut X = self.initial_positions();

        // First layout all nodes to get correct relative positions of all nodes
        let mut prev_stress = stress(&X, &w, &d);
        let Lw = construct_Lw(&w).submatrix(1, w.dim.0);
        loop {
            let LZ = construct_LZ(&X, &alpha);
            for i in 0..2 {
                let b = Vector {
                    inner: (1..LZ.dim.0)
                        .map(|j| (0..LZ.dim.1).map(|k| LZ.data[j][k] * X[k][i]).sum::<f64>())
                        .collect_vec(),
                };
                let x = linear_solve(&Lw, &b, None, None);
                x.inner.into_iter().enumerate().for_each(|(j, x)| X[j + 1][i] = x);
            }
            let current_stress = stress(&X, &w, &d);
            if current_stress == 0.0 || prev_stress - current_stress < 1e-4 * prev_stress {
                break;
            }
            prev_stress = current_stress;
        }

        if n_ext < 2 {
            return NodeLayout { node_positions: X };
        }

        // With correct relative placement, layout the external nodes again without considering internal nodes
        prev_stress = sub_stress(&X, &w, &d, 0, n_ext);
        let Lw = construct_Lw(&w.submatrix(0, n_ext)).submatrix(1, n_ext);
        loop {
            let LZ = construct_LZ(&X[..n_ext], &alpha.submatrix(0, n_ext));
            for i in 0..2 {
                let b = Vector {
                    inner: (1..n_ext)
                        .map(|j| (0..n_ext).map(|k| LZ.data[j][k] * X[k][i]).sum::<f64>())
                        .collect_vec(),
                };
                let x = linear_solve(&Lw, &b, None, None);
                x.inner.into_iter().enumerate().for_each(|(j, x)| X[j + 1][i] = x);
            }
            let current_stress = sub_stress(&X, &w, &d, 0, n_ext);
            if current_stress == 0.0 || prev_stress - current_stress < 1e-4 * prev_stress {
                break;
            }
            prev_stress = current_stress;
        }

        // Keep external nodes fixed and do final layout of only the internal nodes
        prev_stress = stress(&X, &w, &d);
        let Lw = construct_Lw(&w);
        loop {
            let LZ = construct_LZ(&X, &alpha);
            for i in 0..2 {
                let b = Vector {
                    inner: (n_ext..LZ.dim.0)
                        .map(|j| {
                            (0..LZ.dim.1).map(|k| LZ.data[j][k] * X[k][i]).sum::<f64>()
                                - (1..n_ext).map(|k| Lw[j][k] * X[k][i]).sum::<f64>()
                        })
                        .collect_vec(),
                };
                let x = linear_solve(&Lw.submatrix(n_ext, w.dim.0), &b, None, None);
                x.inner.into_iter().enumerate().for_each(|(j, x)| X[j + n_ext][i] = x);
            }
            let current_stress = stress(&X, &w, &d);
            if current_stress == 0.0 || prev_stress - current_stress < 1e-4 * prev_stress {
                break;
            }
            prev_stress = current_stress;
        }

        // Move origin to average position of incoming nodes $\vec{r}_\mathrm{in}$ and align connection vector of
        // incoming and outgoing nodes $\vec{r}_\mathrmc{out} with $x$-axis
        let r_in = X.iter().take(self.n_in()).sum::<Vec2D>() / (self.n_in() as f64);
        X = X.iter().map(|x| *x - r_in).collect_vec();
        let r_out = X.iter().skip(self.n_in()).take(self.n_out()).sum::<Vec2D>() / (self.n_out() as f64);
        let theta = (r_out.y).atan2(r_out.x);
        let (sin, cos) = theta.sin_cos();
        X = X
            .iter()
            .map(|v| Vec2D {
                x: cos * v.x + sin * v.y,
                y: -sin * v.x + cos * v.y,
            })
            .collect();

        return NodeLayout { node_positions: X };
    }
}

#[derive(Debug)]
pub(crate) struct NodeLayout {
    node_positions: Vec<Vec2D>,
}

impl Deref for NodeLayout {
    type Target = Vec<Vec2D>;
    fn deref(&self) -> &Self::Target {
        return &self.node_positions;
    }
}

/// Stress function according to equation (2),
/// $$ \mathrm{stress}(X; w, d, \Gamma) = \sum_{i<j}w_{ij}\left(\left|X_i - X_j\right| - d_{ij}\right)^2 $$
fn stress(X: &[Vec2D], w: &Matrix, d: &Matrix) -> f64 {
    (0..X.len())
        .map(|i| {
            (i + 1..X.len())
                .map(|j| {
                    let xij_diff = (X[i] - X[j]).sq_norm().sqrt() - d[i][j];
                    w[i][j] * xij_diff * xij_diff
                })
                .sum::<f64>()
        })
        .sum::<f64>()
}

/// Stress function according to equation (2) restricted to a submatrix with rows $n$ to $m$,
/// $$ \mathrm{stress}(X; w, d, \Gamma) = \sum_{n < i<j < m}w_{ij}\left(\left|X_i - X_j\right| - d_{ij}\right)^2 $$
fn sub_stress(X: &[Vec2D], w: &Matrix, d: &Matrix, n: usize, m: usize) -> f64 {
    (n..m)
        .map(|i| {
            (i + 1..m)
                .map(|j| {
                    let xij_diff = (X[i] - X[j]).sq_norm().sqrt() - d[i][j];
                    w[i][j] * xij_diff * xij_diff
                })
                .sum::<f64>()
        })
        .sum::<f64>()
}

/// Weight matrix, $$ L_{ij}^w = \begin{cases}-w_{ij} & i \neq j \\ \sum_{k\neq i} w_{ik} & i = j\end{cases} $$
/// Since $w_{ii} = 0$ here, the sum can run over all values of $k$.
fn construct_Lw(w: &Matrix) -> Matrix {
    Matrix {
        dim: w.dim,
        data: (0..w.dim.0)
            .map(|i| {
                let colsum = w[i].iter().sum::<f64>();
                (0..w.dim.1)
                    .map(|j| if i != j { -w[i][j] } else { colsum })
                    .collect_vec()
            })
            .collect_vec(),
    }
}

/// Bounding matrix, $$ L_{ij}^w = \begin{cases}-\beta_{ij} & i \neq j \\ \sum_{k\neq i} \beta_{ik} & i = j\end{cases} $$
/// where $$ \beta_{ij} = \begin{cases} - \frac{\alpha_{ij}}{|Z_i - Z_j|} & i \neq j  \\ 0 & i = j\end{cases}$$
fn construct_LZ(Z: &[Vec2D], alpha: &Matrix) -> Matrix {
    let beta = (0..Z.len())
        .map(|i| {
            (0..Z.len())
                .map(|j| {
                    if i == j {
                        0.
                    } else {
                        -alpha[i][j] / (Z[i] - Z[j]).sq_norm().sqrt()
                    }
                })
                .collect_vec()
        })
        .collect_vec();
    Matrix {
        dim: (beta.len(), beta.len()),
        data: (0..beta.len())
            .map(|i| {
                let colsum = -beta[i].iter().sum::<f64>();
                (0..beta.len())
                    .map(|j| if i != j { beta[i][j] } else { colsum })
                    .collect_vec()
            })
            .collect_vec(),
    }
}

/// Solve the linear system $Ax = b$ to the given precision $\varepsilon$ with the BiCGSTAB algorithm
fn linear_solve(A: &Matrix, b: &Vector, x0: Option<Vector>, eps: Option<f64>) -> Vector {
    let mut x = x0.unwrap_or(Vector {
        inner: vec![0.; A.dim.1],
    });
    if b.inner.iter().all(|x| *x == 0.0) {
        return x;
    }
    let eps = eps.unwrap_or(1e-8);
    let mut r = b.clone() - A * &x;
    let rhat = r.clone();
    let mut p = r.clone();
    let mut v;
    let mut rho = rhat.sq_norm();
    let mut alpha;
    let mut omega;
    loop {
        v = A * &p;
        alpha = rho / (&rhat * &v);
        let h = x.clone() + alpha * p.clone();
        let s = r.clone() - alpha * &v;
        if s.sq_norm() < eps {
            x = h;
            break;
        }
        let t = A * &s;
        omega = (&t * &s) / t.sq_norm();
        x = h + omega * s.clone();
        r = s - omega * t;
        if r.sq_norm() < eps {
            break;
        }
        let new_rho = &rhat * &r;
        p = r.clone() + (new_rho / rho) * (alpha / omega) * (p - omega * v);
        rho = new_rho;
    }
    if cfg!(debug_assertions) {
        (A * &x).inner.iter().zip(b.inner.iter()).for_each(|(x, y)| {
            if (*x - *y).abs() >= eps.sqrt() {
                println!("Invalid linear system solution: Ax_i = {}, b_i = {}", x, y)
            };
            assert!((*x - *y).abs() < eps.sqrt())
        });
    }
    return x;
}

#[cfg(test)]
mod tests {
    use super::super::math::{Matrix, Vec2D};
    use super::*;
    use crate::diagram::{Leg, Propagator, Vertex};
    use crate::model::TopologyModel;
    use crate::topology::TopologyGenerator;
    use crate::topology::{Edge, Node, Topology, components::NodeClassification};
    use approx::assert_relative_eq;

    #[test]
    fn linear_solve_test() {
        let mat = Matrix {
            dim: (2, 2),
            data: vec![vec![4., 1.], vec![1., 3.]],
        };
        let rhs = Vector { inner: vec![1., 2.] };
        assert_relative_eq!(
            linear_solve(&mat, &rhs, None, None).inner.as_slice(),
            vec![1. / 11., 7. / 11.].as_slice()
        )
    }

    #[test]
    fn stress_function_test() {
        let d = Matrix {
            dim: (6, 6),
            data: vec![
                vec![0., 2., 1., 2., 3., 3.],
                vec![2., 0., 1., 2., 3., 3.],
                vec![1., 1., 0., 1., 2., 2.],
                vec![2., 2., 1., 0., 1., 1.],
                vec![3., 3., 2., 1., 0., 2.],
                vec![3., 3., 2., 1., 2., 0.],
            ],
        };
        let w = d.map(|x| if *x != 0. { 1. / ((*x) * (*x)) } else { 0. });
        let x = vec![
            Vec2D { x: 0., y: 0. },
            Vec2D { x: 0., y: 2. },
            Vec2D { x: 1., y: 1. },
            Vec2D { x: 2., y: 1. },
            Vec2D { x: 3., y: 0. },
            Vec2D { x: 3., y: 2. },
        ];
        assert_relative_eq!(stress(&x, &w, &d), 0.82350677928631732201, max_relative = 1e-6);
    }

    #[test]
    fn Lw_test() {
        let d = Matrix {
            dim: (6, 6),
            data: vec![
                vec![0., 2., 1., 2., 3., 3.],
                vec![2., 0., 1., 2., 3., 3.],
                vec![1., 1., 0., 1., 2., 2.],
                vec![2., 2., 1., 0., 1., 1.],
                vec![3., 3., 2., 1., 0., 2.],
                vec![3., 3., 2., 1., 2., 0.],
            ],
        };
        let w = d.map(|x| if *x != 0. { 1. / ((*x) * (*x)) } else { 0. });
        let Lw = construct_Lw(&w);

        let ref_mat = Matrix {
            dim: (6, 6),
            data: vec![
                vec![31. / 18., -1. / 4., -1., -1. / 4., -1. / 9., -1. / 9.],
                vec![-1. / 4., 31. / 18., -1., -1. / 4., -1. / 9., -1. / 9.],
                vec![-1., -1., 7. / 2., -1., -1. / 4., -1. / 4.],
                vec![-1. / 4., -1. / 4., -1., 7. / 2., -1., -1.],
                vec![-1. / 9., -1. / 9., -1. / 4., -1., 31. / 18., -1. / 4.],
                vec![-1. / 9., -1. / 9., -1. / 4., -1., -1. / 4., 31. / 18.],
            ],
        };

        for i in 0..5 {
            for j in 0..5 {
                assert_relative_eq!(Lw[i][j], ref_mat[i][j], max_relative = 1e-7)
            }
        }
    }

    #[test]
    fn LZ_test() {
        let Z = vec![
            Vec2D { x: 0., y: 0. },
            Vec2D { x: 0., y: 2. },
            Vec2D { x: 1., y: 1. },
            Vec2D { x: 2., y: 1. },
            Vec2D { x: 3., y: 0. },
            Vec2D { x: 3., y: 2. },
        ];
        let d = Matrix {
            dim: (6, 6),
            data: vec![
                vec![0., 2., 1., 2., 3., 3.],
                vec![2., 0., 1., 2., 3., 3.],
                vec![1., 1., 0., 1., 2., 2.],
                vec![2., 2., 1., 0., 1., 1.],
                vec![3., 3., 2., 1., 0., 2.],
                vec![3., 3., 2., 1., 2., 0.],
            ],
        };
        let alpha = d.map(|x| if *x != 0. { 1. / *x } else { 0. });
        let LZ = construct_LZ(&Z, &alpha);

        let LZ_ref = Matrix {
            dim: (6, 6),
            data: vec![
                vec![
                    1.3842747,
                    -0.25000000,
                    -0.70710678,
                    -0.22360680,
                    -0.11111111,
                    -0.092450033,
                ],
                vec![
                    -0.25000000,
                    1.3842747,
                    -0.70710678,
                    -0.22360680,
                    -0.092450033,
                    -0.11111111,
                ],
                vec![
                    -0.70710678,
                    -0.70710678,
                    2.8614272,
                    -1.0000000,
                    -0.22360680,
                    -0.22360680,
                ],
                vec![
                    -0.22360680,
                    -0.22360680,
                    -1.0000000,
                    2.8614272,
                    -0.70710678,
                    -0.70710678,
                ],
                vec![
                    -0.11111111,
                    -0.092450033,
                    -0.22360680,
                    -0.70710678,
                    1.3842747,
                    -0.25000000,
                ],
                vec![
                    -0.092450033,
                    -0.11111111,
                    -0.22360680,
                    -0.70710678,
                    -0.25000000,
                    1.3842747,
                ],
            ],
        };

        for i in 0..6 {
            for j in 0..6 {
                assert_relative_eq!(LZ[i][j], LZ_ref[i][j], max_relative = 1e-6)
            }
        }
    }

    #[test]
    fn diag_layout_test() {
        let diag = Diagram {
            incoming_legs: vec![
                Leg {
                    vertex: 0,
                    particle: 0,
                    momentum: vec![],
                },
                Leg {
                    vertex: 0,
                    particle: 0,
                    momentum: vec![],
                },
            ],
            outgoing_legs: vec![
                Leg {
                    vertex: 1,
                    particle: 0,
                    momentum: vec![],
                },
                Leg {
                    vertex: 1,
                    particle: 0,
                    momentum: vec![],
                },
            ],
            propagators: vec![Propagator {
                vertices: [0, 1],
                particle: 0,
                momentum: vec![],
            }],
            vertices: vec![
                Vertex {
                    propagators: vec![-4, -3, 0],
                    interaction: 0,
                },
                Vertex {
                    propagators: vec![-2, -1, 0],
                    interaction: 0,
                },
            ],
            vertex_symmetry: 0,
            propagator_symmetry: 0,
            bridges: vec![],
            sign: 1,
        };
        let mut layout = DiagramLayout {
            diag: &diag,
            distances: None,
        };
        layout.calculate_distances();
        let d_ref = vec![
            vec![0, 2, 3, 3, 1, 2],
            vec![2, 0, 3, 3, 1, 2],
            vec![3, 3, 0, 2, 2, 1],
            vec![3, 3, 2, 0, 2, 1],
            vec![1, 1, 2, 2, 0, 1],
            vec![2, 2, 1, 1, 1, 0],
        ];
        assert_eq!(d_ref, *layout.distances.as_ref().unwrap());

        let positions = layout.layout();
        for v in positions.node_positions.iter() {
            println!("{{{}, {}}}", v.x, v.y);
        }
    }

    #[test]
    fn topo_layout_test() {
        let topo = Topology {
            n_external: 4,
            n_loops: 0,
            nodes: vec![
                Node {
                    degree: 1,
                    adjacent_nodes: vec![4],
                },
                Node {
                    degree: 1,
                    adjacent_nodes: vec![4],
                },
                Node {
                    degree: 1,
                    adjacent_nodes: vec![4],
                },
                Node {
                    degree: 1,
                    adjacent_nodes: vec![4],
                },
                Node {
                    degree: 4,
                    adjacent_nodes: vec![0, 1, 2, 3],
                },
            ],
            edges: vec![
                Edge {
                    connected_nodes: [0, 4],
                    momenta: Some(vec![1, 0, 0, 0]),
                },
                Edge {
                    connected_nodes: [1, 4],
                    momenta: Some(vec![0, 1, 0, 0]),
                },
                Edge {
                    connected_nodes: [2, 4],
                    momenta: Some(vec![0, 0, 1, 0]),
                },
                Edge {
                    connected_nodes: [3, 4],
                    momenta: Some(vec![0, 0, 0, 1]),
                },
            ],
            node_symmetry: 1,
            edge_symmetry: 1,
            momentum_labels: vec!["p1".into(), "p2".into(), "p3".into(), "p4".into()],
            bridges: vec![],
            node_classification: NodeClassification {
                boundaries: vec![0, 1, 2, 3, 4, 5],
                matrix: vec![
                    vec![0, 0, 0, 0, 1],
                    vec![0, 0, 0, 0, 1],
                    vec![0, 0, 0, 0, 1],
                    vec![0, 0, 0, 0, 1],
                    vec![1, 1, 1, 1, 0],
                ],
            },
        };

        let layout = TopologyLayout::from(&topo).layout();
        println!("{:#?}", layout);
    }

    #[test]
    fn propagator_layout_test() {
        let topos = TopologyGenerator::new(2, 1, TopologyModel::from(vec![3, 4]), None).generate();
        TopologyLayout::from(&topos[0]).layout();
        assert!(true);
    }

    #[test]
    fn vacuum_bubble_layout_test() {
        let topos = TopologyGenerator::new(0, 2, TopologyModel::from(vec![3, 4]), None).generate();
        TopologyLayout::from(&topos[0]).layout();
        assert!(true);
    }
}
