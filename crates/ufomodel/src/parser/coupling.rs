use std::collections::HashMap;

pub(crate) enum RawCouplingValue {
    Finite(String),
    Expansion(HashMap<i64, String>),
}
