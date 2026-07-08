use std::{cmp::Ordering, slice};

use crate::{Expression, ExpressionAtom};

impl<A: ExpressionAtom> Expression<A> {
    pub fn normalize(self) -> Self {
        match self {
            x @ (Self::Atom(_) | Self::IConst(_) | Self::FConst(_)) => x,
            Self::Inverse(x) => match *x {
                // Fold multiple inverses [1/(1/x)) -> x]
                Self::Inverse(y) => (*y).normalize(),
                // Shift inverses into numeric primitives
                Self::IConst(i) => Self::FConst((i as f64).recip()),
                Self::FConst(f) => Self::FConst(f.recip()),
                y => Self::Inverse(Box::new(y.normalize())),
            },
            Self::Power(x, e) => Self::Power(Box::new(x.normalize()), Box::new(e.normalize())),
            Self::Function { module, name, exprs } => Self::Function {
                module,
                name,
                exprs: exprs.into_iter().map(|e| e.normalize()).collect(),
            },
            Self::Sum(exprs) => {
                let mut norm_exprs: Vec<Self> = exprs.into_iter().map(|e| e.normalize()).collect();
                norm_exprs.sort_unstable_by_key(|e| e.sort_key_sum());
                let mut result = Vec::new();

                // Fold all consts to a single i64 or f64
                let mut c_i: i64 = 0;
                while let Some(Self::IConst(i)) = norm_exprs.pop_if(|e| match e {
                    Self::IConst(_) => true,
                    _ => false,
                }) {
                    c_i += i;
                }
                let mut c_f: f64 = 0.;
                while let Some(Self::FConst(f)) = norm_exprs.pop_if(|e| match e {
                    Self::FConst(_) => true,
                    _ => false,
                }) {
                    c_f += f;
                }

                // Flatten nested sums and fold inner constants into global constant
                while let Some(Self::Sum(mut inner)) = norm_exprs.pop_if(|e| match e {
                    Self::Sum(_) => true,
                    _ => false,
                }) {
                    match inner.first() {
                        Some(Self::IConst(i)) => {
                            c_i += i;
                            result.extend(inner.drain(1..));
                        }
                        Some(Self::FConst(f)) => {
                            c_f += f;
                            result.extend(inner.drain(1..));
                        }
                        _ => result.append(&mut inner),
                    }
                }

                match (c_i, c_f) {
                    (0, 0.) => (),
                    // Retain integer precision
                    (x, 0.) => result.push(Self::IConst(x)),
                    (0, x) => result.push(Self::FConst(x)),
                    // Coefficient has only floating-point precision overall, so use single float constant
                    (x, y) => result.push(Self::FConst(x as f64 * y)),
                }

                // Move remaining expressions and do final merge + sort
                result.append(&mut norm_exprs);

                if result.is_empty() {
                    Self::IConst(0)
                } else {
                    Self::merge_terms(result)
                }
            }
            Self::Product(exprs) => {
                let mut norm_exprs: Vec<Self> = exprs.into_iter().map(|e| e.normalize()).collect();
                norm_exprs.sort_unstable_by_key(|e| e.sort_key_prod());
                let mut result = Vec::new();

                // Fold all consts to a single i64 or f64
                let mut c_i: i64 = 1;
                while let Some(Self::IConst(i)) = norm_exprs.pop_if(|e| match e {
                    Self::IConst(_) => true,
                    _ => false,
                }) {
                    c_i *= i;
                }
                let mut c_f: f64 = 1.;
                while let Some(Self::FConst(f)) = norm_exprs.pop_if(|e| match e {
                    Self::FConst(_) => true,
                    _ => false,
                }) {
                    c_f *= f;
                }
                // Early exit if the product contains a constant zero factor
                if c_i == 0 || c_f == 0. {
                    return Self::IConst(0);
                }

                // Collect and normalize the denominator
                let mut denom_exprs = Vec::new();
                while let Some(expr) = norm_exprs.pop_if(|e| match e {
                    Self::Inverse(_) => true,
                    _ => false,
                }) {
                    match expr {
                        Self::Inverse(x) => denom_exprs.push(*x),
                        _ => unreachable!(),
                    }
                }
                match denom_exprs.len() {
                    0 => (),
                    1 => result.push(Expression::Inverse(Box::new(denom_exprs.pop().unwrap().normalize()))),
                    _ => result.push(Expression::Inverse(Box::new(
                        Expression::Product(denom_exprs).normalize(),
                    ))),
                };

                // Flatten nested products
                while let Some(Self::Product(mut inner)) = norm_exprs.pop_if(|e| match e {
                    Self::Product(_) => true,
                    _ => false,
                }) {
                    match inner.first() {
                        Some(Self::IConst(i)) => {
                            c_i *= i;
                            result.extend(inner.drain(1..));
                        }
                        Some(Self::FConst(f)) => {
                            c_f *= f;
                            result.extend(inner.drain(1..));
                        }
                        _ => result.append(&mut inner),
                    }
                }

                match (c_i, c_f) {
                    (0, _) => return Self::IConst(0),
                    (_, 0.) => return Self::IConst(0),
                    (1, 1.) => (),
                    (x, 1.) => result.push(Self::IConst(x)),
                    (x, y) => result.push(Self::FConst(x as f64 * y)),
                }

                if let Some(Expression::Sum(_)) = norm_exprs.last() {
                    // Expand the sums in the product
                    let mut expr_buf = norm_exprs.pop().unwrap();
                    while let Some(Self::Sum(inner)) = norm_exprs.pop_if(|e| match e {
                        Self::Sum(_) => true,
                        _ => false,
                    }) {
                        expr_buf = Self::merge_terms(match expr_buf {
                            Self::Sum(x) => x
                                .into_iter()
                                .flat_map(move |expr| {
                                    inner.clone().into_iter().map(move |y| (y * expr.clone()).normalize())
                                })
                                .collect(),
                            _ => unreachable!(),
                        });
                    }
                    // Move remaining expressions and do final sort
                    result.append(&mut norm_exprs);
                    result.sort_unstable_by(|x, y| x.cmp(y));

                    match expr_buf.normalize() {
                        Self::Sum(exprs) => {
                            Self::Sum(exprs.into_iter().map(|e| e * Self::Product(result.clone())).collect())
                        }
                        x => x * Self::Product(result),
                    }
                } else {
                    // Nothing to expand
                    // Move remaining expressions and do final sort
                    result.append(&mut norm_exprs);
                    result.sort_unstable_by(|x, y| x.cmp(y));
                    Self::Product(result)
                }
            }
        }
    }

    fn sort_key_sum(&self) -> usize {
        match self {
            Self::Atom(_) => 1,
            Self::Power(_, _) => 2,
            Self::Function { .. } => 3,
            Self::Product(_) => 4,
            Self::Sum(_) => 5,
            Self::FConst(_) => 6,
            Self::IConst(_) => 7,
            // Inverses can only appear in products
            Self::Inverse(_) => unreachable!(),
        }
    }

    fn sort_key_prod(&self) -> usize {
        match self {
            Self::Atom(_) => 1,
            Self::Power(_, _) => 2,
            Self::Function { .. } => 3,
            Self::Sum(_) => 4,
            Self::Product(_) => 5,
            Self::Inverse(_) => 6,
            Self::FConst(_) => 7,
            Self::IConst(_) => 8,
        }
    }

    pub(crate) fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            // Only ever one constant at this point, so no need for Ordering::Equal or ordering between them
            (Self::IConst(_) | Self::FConst(_), _) => Ordering::Less,
            (Self::Atom(x), Self::Atom(y)) => x.partial_cmp(y).unwrap(),
            (Self::Atom(_), Self::IConst(_) | Self::FConst(_)) => Ordering::Greater,
            (Self::Atom(_), _) => Ordering::Less,
            (Self::Power(..), Self::IConst(_) | Self::FConst(_) | Self::Atom(_)) => Ordering::Greater,
            (Self::Power(x, a), Self::Power(y, b)) => {
                #[allow(irrefutable_let_patterns)]
                if let res = x.cmp(y)
                    && res != Ordering::Equal
                {
                    return res;
                } else if let res = a.cmp(b)
                    && res != Ordering::Equal
                {
                    return res;
                } else {
                    return Ordering::Equal;
                }
            }
            (Self::Power(..), _) => Ordering::Less,
            (Self::Function { .. }, Self::IConst(_) | Self::FConst(_) | Self::Atom(_) | Self::Power(..)) => {
                Ordering::Greater
            }
            (
                Self::Function { module, name, exprs },
                Self::Function {
                    module: module_other,
                    name: name_other,
                    exprs: exprs_other,
                },
            ) => {
                #[allow(irrefutable_let_patterns)]
                if let res = module.cmp(module_other)
                    && res != Ordering::Equal
                {
                    return res;
                } else if let res = name.cmp(name_other)
                    && res != Ordering::Equal
                {
                    return res;
                } else {
                    for (x, y) in exprs.iter().zip(exprs_other.iter()) {
                        if let res = x.cmp(y)
                            && res != Ordering::Equal
                        {
                            return res;
                        }
                    }
                }
                return Ordering::Equal;
            }
            (Self::Function { .. }, _) => Ordering::Less,
            // Lexicographical ordering of the inner terms
            (Self::Sum(x), Self::Sum(y)) => {
                match x.len().cmp(&y.len()) {
                    Ordering::Equal => (),
                    x => return x,
                }
                for (a, b) in x.iter().zip(y.iter()) {
                    match a.cmp(b) {
                        Ordering::Equal => (),
                        x => return x,
                    }
                }
                return Ordering::Equal;
            }
            (Self::Sum(_), Self::IConst(_) | Self::FConst(_) | Self::Atom(_)) => Ordering::Greater,
            (Self::Sum(_), _) => Ordering::Less,
            // Lexicographical ordering of the inner terms
            (Self::Product(x), Self::Product(y)) => {
                let x_iter = x.iter().skip_while(|e| match **e {
                    Self::IConst(_) | Self::FConst(_) => true,
                    _ => false,
                });
                let y_iter = y.iter().skip_while(|e| match **e {
                    Self::IConst(_) | Self::FConst(_) => true,
                    _ => false,
                });
                match x_iter.clone().count().cmp(&y_iter.clone().count()) {
                    Ordering::Equal => (),
                    x => return x,
                }
                for (a, b) in x_iter.zip(y_iter) {
                    match a.cmp(b) {
                        Ordering::Equal => (),
                        x => return x,
                    }
                }
                return Ordering::Equal;
            }
            (Self::Product(_), Self::IConst(_) | Self::FConst(_) | Self::Atom(_) | Self::Sum(_)) => Ordering::Greater,
            (Self::Product(_), _) => Ordering::Less,
            // Only ever one Inverse at this point
            (Self::Inverse(_), _) => Ordering::Greater,
        }
    }

    fn peek_factors(&self) -> &[Self] {
        match self {
            Self::Product(x) => match x.first() {
                Some(Self::IConst(_) | Self::FConst(_)) => return &x[1..],
                Some(_) => return &x[..],
                None => unreachable!(),
            },
            Self::IConst(_) | Self::FConst(_) => return &[],
            x => slice::from_ref(x),
        }
    }

    fn take_factors(self) -> Option<Self> {
        match self {
            Self::Product(mut x) => match x.first() {
                Some(Self::IConst(_) | Self::FConst(_)) => {
                    x.remove(0);
                    return Some(Self::Product(x));
                }
                Some(_) => return Some(Self::Product(x)),
                None => unreachable!(),
            },
            Self::IConst(_) | Self::FConst(_) => return None,
            x => return Some(x),
        }
    }

    fn take_const(&self) -> Option<Self> {
        match self {
            Self::Product(x) => match x.first() {
                Some(expr @ (Self::IConst(_) | Self::FConst(_))) => return Some(expr.clone()),
                Some(_) => return None,
                None => unreachable!(),
            },
            x @ (Self::IConst(_) | Self::FConst(_)) => Some(x.clone()),
            _ => None,
        }
    }

    fn merge_terms(mut terms: Vec<Self>) -> Self {
        terms.sort_unstable_by(|x, y| x.cmp(y));
        let mut result = Vec::with_capacity(terms.len());
        let mut c_i: i64 = 0;
        let mut c_f: f64 = 0.;
        let mut current: Self = Self::IConst(0);
        for (k, expr) in terms.into_iter().enumerate() {
            if k == 0 || current.peek_factors() == expr.peek_factors() {
                match expr.take_const() {
                    Some(Self::IConst(i)) => c_i += i,
                    Some(Self::FConst(f)) => c_f += f,
                    None => c_i += 1,
                    Some(_) => unreachable!(),
                }
                if k == 0 {
                    current = expr;
                }
            } else {
                Self::push_summand(c_i, c_f, current, &mut result);
                c_i = 0;
                c_f = 0.;
                match expr.take_const() {
                    Some(Self::IConst(i)) => c_i += i,
                    Some(Self::FConst(f)) => c_f += f,
                    None => c_i += 1,
                    Some(_) => unreachable!(),
                }
                current = expr;
            }
        }
        Self::push_summand(c_i, c_f, current, &mut result);
        match result.len() {
            0 => Self::IConst(0),
            1 => result.pop().unwrap(),
            _ => {
                result.shrink_to_fit();
                result.sort_unstable_by(|x, y| x.cmp(y));
                return Self::Sum(result);
            }
        }
    }

    fn push_summand(c_i: i64, c_f: f64, expr: Self, buffer: &mut Vec<Self>) {
        match (c_i, c_f, expr.take_factors()) {
            (0, 0., _) => (),
            (i, 0., None) => buffer.push(Self::IConst(i)),
            (1, 0., Some(e)) => buffer.push(e),
            (i, 0., Some(e)) => buffer.push((Self::IConst(i) * e).normalize()),
            (i, f, None) => buffer.push(Self::FConst(i as f64 + f)),
            (i, f, Some(e)) if i as f64 + f == 0. => buffer.push(e),
            (i, f, Some(e)) => buffer.push((Self::FConst(i as f64 + f) * e).normalize()),
        }
    }
}
