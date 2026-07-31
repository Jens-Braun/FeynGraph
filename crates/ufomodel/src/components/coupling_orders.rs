#[derive(Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CouplingOrder {
    pub(crate) name: String,
    pub(crate) expansion_order: usize,
    pub(crate) hierarchy: usize,
    pub(crate) perturbative_expansion: usize,
}
