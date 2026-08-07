use crate::Amplitude;
use expression::{Expression, ExpressionAtom};
use model::analytic::{AnalyticModel, ChainIndices};

impl<M: AnalyticModel> Amplitude<M>
where
    M::Atom: ChainIndices,
{
    pub fn sort_chains(&mut self) {
        self.numerator.sort_chains()
    }
}

trait SortChains {
    fn sort_chains(&mut self);
}
impl<A: ExpressionAtom + ChainIndices> SortChains for Expression<A> {
    fn sort_chains(&mut self) {
        match self {
            Expression::Product(exprs) => {
                let pair_options = exprs
                    .iter()
                    .enumerate()
                    .map(|(i, expr)| (i, expr.chain_indices()))
                    .collect::<Vec<_>>();
                // No spin indices, put at front of expression
                let mut chains = pair_options
                    .iter()
                    .filter_map(|(i, x)| if x.is_none() { Some(*i) } else { None })
                    .collect::<Vec<_>>();
                let mut pairs = pair_options
                    .iter()
                    .filter_map(|(i, x)| x.map(|ind| (*i, ind)))
                    .collect::<Vec<_>>();
                // Collect external chains
                while let Some(i) = pairs.iter().position(|(_, (i, _))| *i == 0) {
                    let (i, (_, start)) = pairs.swap_remove(i);
                    chains.push(i);
                    let mut current = start;
                    while let Some(j) = pairs.iter().position(|(_, (k, l))| *k == current || *l == current) {
                        let (j, (k, l)) = pairs.swap_remove(j);
                        chains.push(j);
                        let next = if k == current { l } else { k };
                        current = next;
                        if next == 0 {
                            break;
                        }
                    }
                }
                // Collect remaining spin pairs, which must build internal chains
                while let Some((i, (start, mut current))) = pairs.pop() {
                    chains.push(i);
                    while let Some(j) = pairs.iter().position(|(_, (k, l))| *k == current || *l == current) {
                        let (j, (k, l)) = pairs.swap_remove(j);
                        chains.push(j);
                        let next = if k == current { l } else { k };
                        current = next;
                        if next == start {
                            break;
                        }
                    }
                }
                *exprs = chains.into_iter().map(|i| exprs[i].clone()).collect::<Vec<_>>();
            }
            Expression::Sum(exprs) => exprs.iter_mut().for_each(|e| e.sort_chains()),
            _ => (),
        }
    }
}
