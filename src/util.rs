use itertools::Itertools;
pub(crate) fn generate_permutations(partition_sizes: &[usize]) -> impl Iterator<Item = Vec<usize>> {
    return partition_sizes.iter().map(
        |size| {
            (1..=*size).permutations(*size)
        }
    ).multi_cartesian_product().map(|x| x.into_iter().flatten().collect());
}

pub(crate) fn factorial(n: usize) -> usize {
    return (2..=n).product();
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    pub fn permutation_test() {
        let partition_sizes = vec![4, 3, 4, 2];
        let permutations = generate_permutations(&partition_sizes).collect_vec();
        assert_eq!(permutations.len(), partition_sizes.iter().map(
            |x| (2..=*x).product::<usize>()
        ).product::<usize>()
        );
    }
}