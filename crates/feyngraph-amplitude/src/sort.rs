use crate::{AmplitudeAtom, AmpltiudeExpression, ExternalAtom};
use ufomodel::components::LorentzAtom;

pub trait SortSpinChains {
    fn spin_indices(&self) -> Option<(i64, i64)>;
    fn sort_chains(&mut self);
}

impl SortSpinChains for AmpltiudeExpression {
    fn spin_indices(&self) -> Option<(i64, i64)> {
        match self {
            Self::Atom(a) => match a {
                AmplitudeAtom::Lorentz(l) => match l {
                    LorentzAtom::Trivial
                    | LorentzAtom::IdentityL(..)
                    | LorentzAtom::Metric(..)
                    | LorentzAtom::P(..)
                    | LorentzAtom::Psq(..)
                    | LorentzAtom::Epsilon(..) => return None,
                    LorentzAtom::Identity(i, j)
                    | LorentzAtom::Gamma(_, i, j)
                    | LorentzAtom::Gamma5(i, j)
                    | LorentzAtom::ProjM(i, j)
                    | LorentzAtom::ProjP(i, j)
                    | LorentzAtom::Sigma(.., i, j)
                    | LorentzAtom::C(i, j) => return Some((*i, *j)),
                },
                AmplitudeAtom::External(a) => match a {
                    ExternalAtom::U(s, _, _) | ExternalAtom::V(s, _, _) => Some((*s, 0)),
                    ExternalAtom::Ubar(s, _, _) | ExternalAtom::Vbar(s, _, _) => Some((0, *s)),
                    _ => None,
                },
                _ => None,
            },
            Self::Inverse(expr) | Self::Power(expr, _) => expr.spin_indices(),
            Self::Sum(exprs) => {
                let mut indices = exprs.iter().map(|e| e.spin_indices()).collect::<Vec<_>>();
                indices.dedup();
                if indices.len() > 1 {
                    panic!()
                } else {
                    return indices.pop().unwrap();
                }
            }
            Self::Product(_) => panic!(),
            _ => None,
        }
    }

    fn sort_chains(&mut self) {
        match self {
            Self::Product(exprs) => {
                let pair_options = exprs
                    .iter()
                    .enumerate()
                    .map(|(i, expr)| (i, expr.spin_indices()))
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
                    while let Some(j) = pairs
                        .iter()
                        .position(|(_, (k, l))| *k == current || *l == current)
                    {
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
                    while let Some(j) = pairs
                        .iter()
                        .position(|(_, (k, l))| *k == current || *l == current)
                    {
                        let (j, (k, l)) = pairs.swap_remove(j);
                        chains.push(j);
                        let next = if k == current { l } else { k };
                        current = next;
                        if next == start {
                            break;
                        }
                    }
                }
                *exprs = chains.into_iter().map(|i| exprs[i].clone()).collect();
            }
            Self::Sum(exprs) => exprs.iter_mut().for_each(|e| e.sort_chains()),
            _ => (),
        }
    }
}
