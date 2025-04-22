use std::cmp::Ordering;

/// A symmetric matrix of dimension $n$. The full symmetric matrix is represented by the upper
/// triangular part, which is stored in `data` as a flattened array with $\frac{n(n+1)}{2}$ entries.
/// The indices run from $0$ to $n-1$.
#[derive(PartialEq, Clone)]
pub struct SymmetricMatrix {
    /// Dimension of the matrix
    pub dimension: usize,
    /// Flattened upper triangular part of the matrix
    data: Vec<usize>,
}

impl SymmetricMatrix {
    /// Return an $n$-dimensional symmetric matrix with only zeroes as entries.
    #[inline]
    pub fn zero(dimension: usize) -> Self {
        return Self {
            dimension,
            data: vec![0; dimension * (dimension + 1) / 2],
        };
    }

    /// Return the $n$-dimensional identity matrix.
    #[inline]
    pub fn identity(dimension: usize) -> Self {
        let mut data = Vec::with_capacity(dimension * (dimension + 1) / 2);
        for i in 0..dimension {
            data.push(1);
            data.append(&mut vec![0; dimension - i - 1])
        }
        return Self { dimension, data };
    }

    /// Build $n$-dimensional matrix from Vec.
    #[inline]
    pub fn from_vec(dimension: usize, data: Vec<usize>) -> Self {
        assert_eq!(dimension * (dimension + 1) / 2, data.len());
        return Self { dimension, data };
    }

    /// Return element $A_{ij}$, where $i$ and $j$ run from $0$ to $n-1$.
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> &usize {
        return if j >= i {
            &self.data[i * self.dimension + j - i * (i + 1) / 2]
        } else {
            &self.data[j * self.dimension + i - j * (j + 1) / 2]
        };
    }

    /// Return mutable element $A_{ij}$, where $i$ and $j$ run from $0$ to $n-1$.
    #[inline]
    pub fn get_mut(&mut self, i: usize, j: usize) -> &mut usize {
        return if j >= i {
            &mut self.data[i * self.dimension + j - i * (i + 1) / 2]
        } else {
            &mut self.data[j * self.dimension + i - j * (j + 1) / 2]
        };
    }

    /// Compare with self, when the rows and columns are permuted according to `permutation`, where
    /// `permutation` is a permutation of ${1, ..., n}$. Here, the entries of the matrix are
    /// interpreted as digits of a number $X$ with a base $B > a_{ij} \forall i, j$ , i.e.
    /// $X = \sum_{i, j=1}^{n} a_{ij} \times B^{i*n+j}$.
    pub fn cmp_permutation(&self, permutation: &[usize]) -> Ordering {
        for i in 0..self.dimension {
            for j in i..self.dimension {
                match (*self.get(i, j)).cmp(self.get(permutation[i] - 1, permutation[j] - 1)) {
                    Ordering::Less => return Ordering::Less,
                    Ordering::Greater => return Ordering::Greater,
                    Ordering::Equal => (),
                }
            }
        }
        return Ordering::Equal;
    }
}

impl std::fmt::Debug for SymmetricMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        writeln!(f, r"SymmetricMatrix {{")?;
        writeln!(f, "    dimension: {},", self.dimension)?;
        writeln!(f, "    data: [")?;
        for i in 0..self.dimension {
            write!(f, "        ")?;
            for j in 0..self.dimension {
                write!(f, "{:?}  ", self.get(i, j))?;
            }
            writeln!(f)?;
        }
        writeln!(f, "    ]")?;
        writeln!(f, "}}")?;
        return Ok(());
    }
}

#[cfg(test)]
mod test {
    use super::SymmetricMatrix;
    use std::cmp::Ordering;
    use test_log::test;

    #[test]
    fn cmp_test() {
        let matrix = SymmetricMatrix::from_vec(3, vec![1, 6, 3, 2, 4, 5]);
        assert_eq!(matrix.cmp_permutation(&[1, 2, 3]), Ordering::Equal);
        assert_eq!(matrix.cmp_permutation(&[3, 2, 1]), Ordering::Less);
        assert_eq!(matrix.cmp_permutation(&[1, 3, 2]), Ordering::Greater);
    }

    #[test]
    fn fmt_test() {
        let matrix = SymmetricMatrix::from_vec(3, vec![1, 2, 3, 4, 5, 6]);
        let expected = "\
SymmetricMatrix {
    dimension: 3,
    data: [
        1  2  3  
        2  4  5  
        3  5  6  
    ]
}
";
        assert_eq!(expected, format!("{:#?}", matrix));
    }
}
