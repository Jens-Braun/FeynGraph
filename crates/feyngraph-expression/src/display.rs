use std::fmt::Display;

use crate::{Expression, ExpressionAtom};

impl<A: ExpressionAtom + Display> Display for Expression<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IConst(x) => write!(f, "{}", x),
            Self::FConst(x) => write!(f, "{}", x),
            Self::Atom(a) => write!(f, "{}", a),
            Self::Power(x, e) => write!(f, "{}^({})", x, e),
            Self::Function {
                module,
                name,
                exprs,
            } => {
                if let Some(m) = module {
                    write!(f, "{}.", m)?;
                }
                write!(
                    f,
                    "{}({})",
                    name,
                    exprs
                        .iter()
                        .map(|e| e.to_string())
                        .collect::<Vec<_>>()
                        .join(",")
                )
            }
            Self::Sum(exprs) => {
                for (i, e) in exprs.iter().enumerate() {
                    if i != 0 {
                        write!(f, " + ")?;
                    }
                    write!(f, "{}", e)?;
                }
                Ok(())
            }
            Self::Product(exprs) => {
                for (i, e) in exprs.iter().enumerate() {
                    if i != 0 {
                        match e {
                            Self::Inverse(_) => write!(f, " / ")?,
                            _ => write!(f, " * ")?,
                        }
                    }
                    match e {
                        e @ Self::Sum(_) => write!(f, "({})", e)?,
                        _ => write!(f, "{}", e)?,
                    }
                }
                Ok(())
            }
            Self::Inverse(e) => {
                match &**e {
                    e @ (Self::Sum(_) | Self::Product(_)) => write!(f, "({})", e)?,
                    _ => write!(f, "{}", e)?,
                }
                Ok(())
            }
        }
    }
}
