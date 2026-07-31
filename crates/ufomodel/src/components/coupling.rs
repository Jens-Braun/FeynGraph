use util::HashMap;

use expression::Expression;

use crate::components::ParameterAtom;

pub type CouplingExpression = Expression<ParameterAtom>;

#[derive(Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CouplingValue {
    Finite(CouplingExpression),
    Expansion(HashMap<i64, CouplingExpression>),
}

#[derive(Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Coupling {
    pub name: String,
    pub value: CouplingValue,
    pub order: HashMap<String, usize>,
}
