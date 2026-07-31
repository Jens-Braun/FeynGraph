use util::HashMap;

use serde::{Deserializer, Serializer, de::Visitor, ser::SerializeMap};

pub(crate) fn serialize_coupling_map<S>(map: &HashMap<(usize, usize), String>, ser: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let mut map_ser = ser.serialize_map(Some(map.len()))?;
    for ((c, s), v) in map.iter() {
        map_ser.serialize_entry(&format!("({c},{s})"), v)?;
    }
    return map_ser.end();
}

pub(crate) fn serialize_ct_coupling_map<S>(
    map: &HashMap<(usize, usize, usize), String>,
    ser: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let mut map_ser = ser.serialize_map(Some(map.len()))?;
    for ((c, s, l), v) in map.iter() {
        map_ser.serialize_entry(&format!("({c},{s},{l})"), v)?;
    }
    return map_ser.end();
}

pub(crate) fn deserialize_coupling_map<'de, D>(der: D) -> Result<HashMap<(usize, usize), String>, D::Error>
where
    D: Deserializer<'de>,
{
    struct CouplingMapVisitor;
    impl<'de> Visitor<'de> for CouplingMapVisitor {
        type Value = HashMap<(usize, usize), String>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("a coupling map")
        }

        fn visit_map<A>(self, mut access: A) -> Result<Self::Value, A::Error>
        where
            A: serde::de::MapAccess<'de>,
        {
            let mut map: HashMap<(usize, usize), String> = HashMap::default();
            while let Some((k, v)) = access.next_entry::<String, String>()? {
                let indices = k[1..k.len() - 1]
                    .split(",")
                    .map(|x| x.parse::<usize>())
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap();
                assert!(indices.len() == 2);
                map.insert((indices[0], indices[1]), v);
            }
            return Ok(map);
        }
    }
    der.deserialize_map(CouplingMapVisitor)
}

pub(crate) fn deserialize_ct_coupling_map<'de, D>(der: D) -> Result<HashMap<(usize, usize, usize), String>, D::Error>
where
    D: Deserializer<'de>,
{
    struct CTCouplingMapVisitor;
    impl<'de> Visitor<'de> for CTCouplingMapVisitor {
        type Value = HashMap<(usize, usize, usize), String>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("a coupling map")
        }

        fn visit_map<A>(self, mut access: A) -> Result<Self::Value, A::Error>
        where
            A: serde::de::MapAccess<'de>,
        {
            let mut map: HashMap<(usize, usize, usize), String> = HashMap::default();
            while let Some((k, v)) = access.next_entry::<String, String>()? {
                let indices = k[1..k.len() - 1]
                    .split(",")
                    .map(|x| x.parse::<usize>())
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap();
                assert!(indices.len() == 3);
                map.insert((indices[0], indices[1], indices[2]), v);
            }
            return Ok(map);
        }
    }
    der.deserialize_map(CTCouplingMapVisitor)
}
