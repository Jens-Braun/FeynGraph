use indexmap::IndexMap as OriginalIndexMap;
use itertools::{Itertools, izip};
use rustc_hash::{FxBuildHasher, FxHashMap};
use thiserror::Error;

pub(crate) type IndexMap<K, V> = OriginalIndexMap<K, V, FxBuildHasher>;
pub(crate) type HashMap<K, V> = FxHashMap<K, V>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("Invalid input: {0}")]
    InputError(String),
}

pub(crate) fn generate_permutations(partition_sizes: &[usize]) -> impl Iterator<Item = Vec<usize>> {
    return izip!(
        partition_sizes.iter().scan(1usize, |state, size| {
            let start = *state;
            *state += *size;
            return Some(start);
        }),
        partition_sizes.iter()
    )
    .map(|(start, size)| (start..(start + *size)).permutations(*size))
    .multi_cartesian_product()
    .map(|x| x.into_iter().flatten().collect());
}

pub(crate) fn factorial(n: usize) -> usize {
    return (2..=n).product();
}

/// Find all choices of $N_\nu$ such that $\sum_{\nu=0}^\infty \nu N_\nu = `sum`$
pub(crate) fn find_partitions(values: impl Iterator<Item = usize>, sum: usize) -> Vec<Vec<usize>> {
    let possible_values = values.filter(|x| *x <= sum).collect_vec();
    let max_numbers = possible_values.iter().map(|x| sum / (*x)).collect_vec();
    return max_numbers
        .iter()
        .map(|i| 0..=(*i))
        .multi_cartesian_product()
        .filter(|x| {
            x.iter()
                .enumerate()
                .map(|(i, y)| (*y) * possible_values[i])
                .sum::<usize>()
                == sum
        })
        .collect_vec();
}

/// Contract all internal (negative) indices in a list of connected indices and build a connection map.
/// The `i`-th entry of the connection map contains the index `j` to which index `i` is connected. For indices
/// for which no information is available, the map contains the entry `-1`.
pub(crate) fn contract_indices(mut connections: Vec<(isize, isize)>) -> Vec<isize> {
    if connections.is_empty() {
        return vec![];
    }
    let mut res = vec![
        -1;
        *connections
            .iter()
            .map(|(x, y)| if x > y { x } else { y })
            .max()
            .unwrap() as usize
            + 1
    ];
    while !connections.is_empty() {
        let (start, mut current) = connections.swap_remove(connections.iter().position(|(i, _)| *i >= 0).unwrap());
        while current < 0 {
            let index = connections.iter().position(|(i, _)| *i == current).unwrap();
            current = connections.swap_remove(index).1;
        }
        res[start as usize] = current;
        res[current as usize] = start;
    }
    return res;
}

#[cfg(test)]
mod test {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn permutation_size_test() {
        let partition_sizes = vec![4, 3, 4, 2];
        let permutations = generate_permutations(&partition_sizes).collect_vec();
        assert_eq!(
            permutations.len(),
            partition_sizes
                .iter()
                .map(|x| (2..=*x).product::<usize>())
                .product::<usize>()
        );
    }

    #[test]
    fn permutation_test() {
        let partition_sizes = vec![1, 1, 3];
        let permutations: HashSet<Vec<usize>> = generate_permutations(&partition_sizes).collect();
        let permutations_ref = HashSet::from([
            vec![1, 2, 3, 4, 5],
            vec![1, 2, 4, 3, 5],
            vec![1, 2, 4, 5, 3],
            vec![1, 2, 5, 4, 3],
            vec![1, 2, 5, 3, 4],
            vec![1, 2, 3, 5, 4],
        ]);
        println!(
            "{:#?}",
            izip!(
                partition_sizes.iter().scan(0usize, |state, size| {
                    *state += *size;
                    return Some(*state);
                }),
                partition_sizes.iter()
            )
            .collect_vec()
        );
        assert_eq!(permutations_ref, permutations);
    }

    #[test]
    fn partition_test() {
        let partition_vec = find_partitions([1usize, 2usize].into_iter(), 4);
        let mut partitions = HashSet::new();
        for partition in partition_vec {
            partitions.insert(partition);
        }
        let partitions_ref: HashSet<Vec<usize>> = HashSet::from([vec![0, 2], vec![2, 1], vec![4, 0]]);
        assert_eq!(partitions, partitions_ref);
    }

    #[test]
    fn contract_test() {
        let connections = vec![(0, -1), (1, -3), (-1, -4), (-4, 2), (-3, -5), (-5, 3)];
        assert_eq!(contract_indices(connections), vec![2, 3, 0, 1]);
    }
}
