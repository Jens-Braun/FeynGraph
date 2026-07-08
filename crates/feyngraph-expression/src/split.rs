use super::{Expression, ExpressionAtom};

pub struct SplitExpression<K, A: ExpressionAtom> {
    key_map: Vec<usize>,
    unique_keys: Vec<K>,
    expressions: Vec<Expression<A>>,
}

impl<K, A: ExpressionAtom> SplitExpression<K, A> {
    pub fn expression_key(&self, index: usize) -> &K {
        return &self.unique_keys[self.key_map[index]];
    }

    pub fn expression_keys(&self) -> impl Iterator<Item = &K> {
        return self.key_map.iter().map(|k| &self.unique_keys[*k]);
    }

    pub fn unique_key(&self, index: usize) -> &K {
        return &self.unique_keys[index];
    }

    pub fn unique_keys(&self) -> &[K] {
        return &self.unique_keys;
    }

    pub fn expressions(&self) -> &[Expression<A>] {
        return &self.expressions;
    }

    pub fn into_expressions(self) -> Vec<Expression<A>> {
        return self.expressions;
    }
}

impl<A: ExpressionAtom> Expression<A> {
    pub fn split_by<K: Clone + Ord + PartialEq>(
        self,
        f: impl Fn(&Expression<A>) -> Vec<K>,
    ) -> SplitExpression<K, A> {
        let normalized = self.normalize();
        match normalized {
            expr @ Self::Sum(_) => {
                let keys = f(&expr);
                let mut unique_keys = keys.clone();
                unique_keys.sort();
                unique_keys.dedup();
                let mut expressions = vec![Vec::new(); unique_keys.len()];
                let mut map = Vec::with_capacity(keys.len());
                for (key, e) in keys.iter().zip(match expr {
                    Self::Sum(exprs) => exprs.into_iter(),
                    _ => unreachable!(),
                }) {
                    let i = unique_keys.iter().position(|k| *k == *key).unwrap();
                    map.push(i);
                    expressions[i].push(e);
                }
                return SplitExpression {
                    key_map: map,
                    unique_keys,
                    expressions: expressions
                        .into_iter()
                        .map(|mut exprs| {
                            if exprs.len() > 1 {
                                Self::Sum(exprs)
                            } else {
                                exprs.pop().unwrap()
                            }
                        })
                        .collect(),
                };
            }
            _ => {
                return SplitExpression {
                    key_map: vec![0],
                    unique_keys: f(&normalized),
                    expressions: vec![normalized],
                };
            }
        }
    }
}
