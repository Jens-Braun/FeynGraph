use itertools::Itertools;
use std::{borrow::Borrow, hash::Hash};
use util::HashMap;

pub trait VertexBase {
    fn name(&self) -> &str;
    fn particles(&self) -> &[impl AsRef<str>];
    fn coupling_orders(&self) -> &HashMap<String, usize>;

    fn fermi_map(&self, in_ray: usize) -> usize;

    fn order<Q>(&self, coupling: &Q) -> usize
    where
        Q: Hash + Eq,
        String: Borrow<Q>,
    {
        return *self.coupling_orders().get(coupling).unwrap_or(&0);
    }

    fn degree(&self) -> usize {
        self.particles().len()
    }

    /// Check whether the given particle names match the interaction. "_" can be used as a wildcard to
    /// match all particles.
    fn match_particles<'q, S>(&self, query: impl IntoIterator<Item = &'q S>) -> bool
    where
        S: 'q + PartialEq<str> + Ord,
    {
        let particles_sorted: Vec<&str> = self.particles().iter().map(|s| s.as_ref()).sorted().collect();
        let mut wildcards: usize = 0;
        let query_sorted: Vec<&S> = query
            .into_iter()
            .filter(|s| {
                if *s != "_" {
                    true
                } else {
                    wildcards += 1;
                    false
                }
            })
            .sorted()
            .collect();
        if particles_sorted.len() != wildcards + query_sorted.len() {
            return false;
        }
        let mut query_cursor: usize = 0;
        for p in particles_sorted {
            if query_cursor < query_sorted.len() && *query_sorted[query_cursor] == *p {
                query_cursor += 1;
            } else {
                if wildcards > 0 {
                    wildcards -= 1;
                } else {
                    return false;
                }
            }
        }
        return true;
    }
}
