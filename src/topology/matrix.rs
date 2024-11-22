use std::cmp::Ordering;
use num_traits::Num;

/// A symmetric matrix of dimension $n$. The full symmetric matrix is represented by the upper
/// triangular part, which is stored in `data` as a flattened array with $\frac{n(n+1)}{2}$ entries.
/// The indices run from $0$ to $n-1$.
#[derive(PartialEq, Debug, Clone)]
pub struct SymmetricMatrix<T: Copy + Num + Ord> {
    /// Dimension of the matrix
    pub dimension: usize,
    /// Flattened upper triangular part of the matrix
    data: Vec<T>,
}

impl<T: Copy + Num + Ord> SymmetricMatrix<T> {
    /// Return an $n$-dimensional symmetric matrix with only zeroes as entries.
    #[inline]
    pub fn zero(dimension: usize) -> Self {
        return Self {
            dimension,
            data: vec![T::zero(); dimension*(dimension+1)/2]
        };
    }

    /// Return the $n$-dimensional identity matrix.
    #[inline]
    pub fn identity(dimension: usize) -> Self {
        let mut data = Vec::with_capacity(dimension*(dimension+1)/2);
        for i in 0..dimension {
            data.push(T::one());
            data.append(&mut vec![T::zero(); dimension-i-1])
        }
        return Self {
            dimension,
            data
        }
    }

    /// Build $n$-dimensional matrix from Vec.
    #[inline]
    pub fn from_vec(dimension: usize, data: Vec<T>) -> Self {
        assert_eq!(dimension * (dimension + 1)/2, data.len());
        return Self { dimension, data };
    }

    /// Return element $A_{ij}$, where $i$ and $j$ run from $0$ to $n-1$.
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> &T {
        return if j >= i {
            &self.data[i * self.dimension + j - i*(i+1)/2]
        } else {
            &self.data[j * self.dimension + i - j*(j+1)/2]
        };
    }

    /// Return mutable element $A_{ij}$, where $i$ and $j$ run from $0$ to $n-1$.
    #[inline]
    pub fn get_mut(&mut self, i: usize, j: usize) -> &mut T {
        return if j >= i {
            &mut self.data[i * self.dimension + j - i*(i+1)/2]
        } else {
            &mut self.data[j * self.dimension + i - j*(j+1)/2]
        };
    }

    /// Swap the $i$-th and $j$-the row and column in place.
    pub fn swap(&mut self, i: usize, j: usize) {
        let max;
        let min;
        if i > j {
            max = i;
            min = j;
        } else {
            max = j;
            min = i;
        }
        self.data.swap(i*(2*self.dimension - i + 1)/2, j*(2*self.dimension - j + 1)/2);
        for k in 0..self.dimension {
            if k == i || k == j { continue; }
            if min < k { // Swap (k, min) <-> (k, min)
                if max < k { // min < max < k
                    self.data.swap(min * self.dimension + k - min*(min+1)/2,
                                   max * self.dimension + k - max*(max+1)/2)
                } else { // min < k < max
                    self.data.swap(min * self.dimension + k - min*(min+1)/2,
                                   k * self.dimension + max - k*(k+1)/2)
                }
            } else { // k < min < max
                self.data.swap(k * self.dimension + min - k*(k+1)/2,
                               k * self.dimension + max - k*(k+1)/2)
            }
        }
    }

    /// Compare with self, when the rows and columns are permuted according to `permutation`, where
    /// `permutation` is a permutation of ${1, ..., n}$. Here, the entries of the matrix are
    /// interpreted as digits of a number $X$ with a base $B > a_{ij} \forall i, j$ , i.e.
    /// $X = \sum_{i, j=1}^{n} a_{ij} \times B^{i*n+j}$.
    pub fn cmp_permutation(&self, permutation: &[usize]) -> Ordering {
        for i in 0..self.dimension {
            for j in i..self.dimension {
                match (*self.get(i, j)).cmp(&self.get(permutation[i]-1, permutation[j]-1)) {
                    Ordering::Less => return Ordering::Less,
                    Ordering::Greater => return Ordering::Greater,
                    Ordering::Equal => (),
                }
            }
        }
        return Ordering::Equal;
    }
}

#[cfg(test)]
mod test {
    use std::cmp::Ordering;
    use super::SymmetricMatrix;
    #[test]
    fn swap_test() {
        let mut matrix = SymmetricMatrix::from_vec(5,
                                                   vec![
                                                       0,  1,  2,  3,  4,
                                                           5,  6,  7,  8,
                                                               9, 10, 11,
                                                                  12, 13,
                                                                      14
                                                   ]);
        matrix.swap(1, 3);
        assert_eq!(matrix, SymmetricMatrix::from_vec(5,
                                                   vec![
                                                       0,  3,  2,  1,  4,
                                                          12, 10,  7, 13,
                                                               9,  6, 11,
                                                                   5,  8,
                                                                      14
                                                   ]));
    }

    #[test]
    fn cmp_test() {
        let matrix = SymmetricMatrix::from_vec(3, vec![1, 6, 3, 2, 4, 5]);
        assert_eq!(matrix.cmp_permutation(&[1, 2, 3]), Ordering::Equal);
        assert_eq!(matrix.cmp_permutation(&[3, 2, 1]), Ordering::Less);
        assert_eq!(matrix.cmp_permutation(&[1, 3, 2]), Ordering::Greater);
    }
}