use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::{Expression, ExpressionAtom};

impl<A: ExpressionAtom> Add<Expression<A>> for Expression<A> {
    type Output = Expression<A>;
    fn add(self, rhs: Expression<A>) -> Self::Output {
        return match (self, rhs) {
            (Expression::Sum(mut x), Expression::Sum(mut y)) => {
                x.append(&mut y);
                Expression::Sum(x)
            }
            (Expression::Sum(mut x), y) => {
                x.push(y);
                Expression::Sum(x)
            }
            (x, Expression::Sum(mut y)) => {
                y.push(x);
                Expression::Sum(y)
            }
            (x, y) => Expression::Sum(vec![x, y]),
        };
    }
}

impl<A: ExpressionAtom> Neg for Expression<A> {
    type Output = Expression<A>;
    fn neg(self) -> Self::Output {
        match self {
            Expression::Product(mut x) => {
                x.push(Self::IConst(-1));
                Expression::Product(x)
            }
            x => Expression::Product(vec![Self::IConst(-1), x]),
        }
    }
}

impl<A: ExpressionAtom> Sub<Expression<A>> for Expression<A> {
    type Output = Expression<A>;
    fn sub(self, rhs: Expression<A>) -> Self::Output {
        self + (-rhs)
    }
}

impl<A: ExpressionAtom> Mul<Expression<A>> for Expression<A> {
    type Output = Expression<A>;
    fn mul(self, rhs: Expression<A>) -> Self::Output {
        return match (self, rhs) {
            (Expression::Product(mut x), Expression::Product(mut y)) => {
                x.append(&mut y);
                Expression::Product(x)
            }
            (Expression::Product(mut x), y) => {
                x.push(y);
                Expression::Product(x)
            }
            (x, Expression::Product(mut y)) => {
                y.push(x);
                Expression::Product(y)
            }
            (Expression::Inverse(x), Expression::Inverse(y)) => Expression::Inverse(Box::new(*x * *y)),
            (x, y) => Expression::Product(vec![x, y]),
        };
    }
}

impl<A: ExpressionAtom> Div<Expression<A>> for Expression<A> {
    type Output = Expression<A>;
    fn div(self, rhs: Expression<A>) -> Self::Output {
        self * Expression::Inverse(Box::new(rhs))
    }
}
