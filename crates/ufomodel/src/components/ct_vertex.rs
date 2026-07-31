use crate::components::vertex::ColorAtom;
use expression::Expression;
use util::HashMap;

#[derive(Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CTVertexType {
    R2,
    UV,
    UVmass,
}

#[derive(Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CTVertex {
    pub(crate) name: String,
    pub(crate) particles: Vec<String>,
    pub(crate) color: Vec<Expression<ColorAtom>>,
    pub(crate) lorentz: Vec<String>,
    #[cfg_attr(
        feature = "serde",
        serde(serialize_with = "crate::serde_util::serialize_ct_coupling_map")
    )]
    #[cfg_attr(
        feature = "serde",
        serde(deserialize_with = "crate::serde_util::deserialize_ct_coupling_map")
    )]
    pub(crate) couplings: HashMap<(usize, usize, usize), String>,
    pub(crate) r#type: CTVertexType,
    pub(crate) loop_particles: Vec<Vec<Vec<String>>>,
}
