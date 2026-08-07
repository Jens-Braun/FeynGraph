pub mod unicode;

use indexmap::IndexMap as OriginalIndexMap;
use rustc_hash::{FxBuildHasher, FxHashMap};

pub type IndexMap<K, V> = OriginalIndexMap<K, V, FxBuildHasher>;
pub type HashMap<K, V> = FxHashMap<K, V>;

/// Contract all internal (negative) indices in a list of connected indices and build a connection map.
/// The `i`-th entry of the connection map contains the index `j` to which index `i` is connected. For indices
/// for which no information is available, the map contains the entry `-1`.
pub fn contract_index_map(mut connections: Vec<(isize, isize)>) -> Vec<isize> {
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

/// Contract all internal (negative) indices in a list of connected indices.
pub fn resolve_negative(connections: &mut Vec<(i64, i64)>) {
    'outer: while let Some(i) = connections.iter().position(|(i, j)| *i < 0 || *j < 0) {
        let (start, mut current) = connections.swap_remove(i);
        while current < 0 {
            match connections.iter().position(|(k, l)| *k == current || *l == current) {
                Some(i) => {
                    let (k, l) = connections.swap_remove(i);
                    if current == k {
                        current = l;
                    } else {
                        current = k;
                    }
                }
                None => continue 'outer,
            }
        }
        connections.push((start, current));
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn contract_test() {
        let connections = vec![(0, -1), (1, -3), (-1, -4), (-4, 2), (-3, -5), (-5, 3)];
        assert_eq!(contract_index_map(connections), vec![2, 3, 0, 1]);
    }
}
