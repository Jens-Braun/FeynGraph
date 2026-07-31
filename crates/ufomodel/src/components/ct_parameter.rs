use util::HashMap;

#[derive(Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CTParameterType {
    Real,
    Complex,
}

#[derive(Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CTParameterValue {
    Finite(String),
    Expansion(HashMap<i64, String>),
}

#[derive(Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CTParameter {
    pub(crate) name: String,
    pub(crate) r#type: CTParameterType,
    pub(crate) value: CTParameterValue,
    pub(crate) texname: String,
}
