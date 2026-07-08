use std::mem::discriminant;

use crate::{Expression, ExpressionAtom};

impl<A: ExpressionAtom> Expression<A> {
    pub fn replace(&mut self, query: Self, replacement: Self) {
        let norm_query = query.normalize();
        self.replace_impl(&norm_query, &replacement);
    }

    fn replace_impl(&mut self, query: &Self, replacement: &Self) {
        match (query, &self) {
            // Product can become a sum after normalization, so this has to be considered explicitly
            (Self::Sum(q_exprs), Self::Product(_) | Self::Sum(_)) => {
                let norm_self = self.clone().normalize();
                match norm_self {
                    Self::Product(_) => (),
                    Self::Sum(mut s_exprs) => {
                        if *q_exprs == s_exprs {
                            *self = replacement.clone();
                            return;
                        } else if s_exprs.len() > q_exprs.len() {
                            // Full expression did not match, but a sub expression still might
                            let matches = s_exprs
                                .windows(q_exprs.len())
                                .into_iter()
                                .enumerate()
                                .filter_map(|(i, s)| if s == q_exprs.as_slice() { Some(i) } else { None })
                                .collect::<Vec<_>>();
                            if !matches.is_empty() {
                                for i in matches.into_iter().rev() {
                                    s_exprs.splice(i..(i + q_exprs.len()), [replacement.clone()]);
                                }
                                *self = Self::Sum(s_exprs);
                                return;
                            }
                        }
                    }
                    _ => unreachable!(),
                };
            }
            (Self::Product(q_exprs), Self::Product(_)) => {
                let norm_self = self.clone().normalize();
                match norm_self {
                    Self::Product(mut s_exprs) => {
                        if *q_exprs == s_exprs {
                            *self = replacement.clone();
                            return;
                        } else if s_exprs.len() > q_exprs.len() {
                            // Full expression did not match, but a sub expression still might
                            let matches = s_exprs
                                .windows(q_exprs.len())
                                .into_iter()
                                .enumerate()
                                .filter_map(|(i, s)| if s == q_exprs.as_slice() { Some(i) } else { None })
                                .collect::<Vec<_>>();
                            if !matches.is_empty() {
                                for i in matches.into_iter().rev() {
                                    s_exprs.splice(i..(i + q_exprs.len()), [replacement.clone()]);
                                }
                                *self = Self::Product(s_exprs);
                                return;
                            }
                        }
                    }
                    _ => unreachable!(),
                };
            }
            _ => {
                if discriminant(self) == discriminant(query) {
                    let norm_self = self.clone().normalize();
                    if norm_self == *query {
                        *self = replacement.clone();
                        return;
                    }
                }
            }
        }
        match self {
            // Atomic expressions would have already been processed if they matched
            Self::Atom(_) | Self::IConst(_) | Self::FConst(_) => (),
            Self::Sum(exprs) | Self::Product(exprs) | Self::Function { exprs, .. } => {
                exprs.iter_mut().for_each(|e| e.replace_impl(query, replacement));
            }
            Self::Inverse(expr) => {
                expr.replace_impl(query, replacement);
            }
            Self::Power(x, e) => {
                x.replace_impl(query, replacement);
                e.replace_impl(query, replacement);
            }
        }
    }
}
