use pyo3::{
    prelude::*,
    types::{PyDict, PyList},
};
use std::{ffi::OsStr, path::Path};

use crate::{
    components::*,
    error::{ReadError, ReadErrorKind},
    parser::{InnerReader, RawParameterValue, UFOReader, coupling::RawCouplingValue},
};
use model::LineStyle;
use util::HashMap;

macro_rules! extract {
    ($py:expr, $field:literal) => {
        match $py.getattr($field) {
            Ok(ref v) => v.extract(),
            Err(e) => Err(e),
        }
    };
    ($py:expr, $field:literal => $subfield:literal) => {
        match $py.getattr($field) {
            Ok(ref v) => match v.getattr($subfield) {
                Ok(ref v) => v.extract(),
                Err(e) => Err(e),
            },
            Err(e) => Err(e),
        }
    };
    ($py:expr, $field:literal, $type:ty) => {
        match $py.getattr($field) {
            Ok(ref v) => v.extract::<$type>(),
            Err(e) => Err(e),
        }
    };
    ($py:expr, $field:literal => $subfield:literal, $type:ty) => {
        match $py.getattr($field) {
            Ok(ref v) => match v.getattr($subfield) {
                Ok(ref v) => v.extract::<$type>(),
                Err(e) => Err(e),
            },
            Err(e) => Err(e),
        }
    };
}

macro_rules! extract_map {
    ($py:expr, $field:literal => $subfield:literal) => {
        $py.getattr($field)?
            .cast_into::<PyList>()?
            .into_iter()
            .map(|p| match p.getattr($subfield) {
                Ok(v) => v.extract(),
                Err(e) => Err(e),
            })
            .collect::<Result<Vec<_>, _>>()
    };
    ($py:expr, $field:literal => $subfield:literal, $type:ty) => {
        $py.getattr($field)?
            .cast_into::<PyList>()?
            .into_iter()
            .map(|p| match p.getattr($subfield) {
                Ok(v) => v.extract::<$type>(),
                Err(e) => Err(e),
            })
            .collect::<Result<Vec<_>, _>>();
    };
}

pub(crate) struct CPythonReader {
    model: Py<PyModule>,
}

impl InnerReader for CPythonReader {
    fn new(model: &Path) -> Result<Self, ReadError> {
        let model = Python::attach(|py| -> PyResult<Py<PyModule>> {
            let syspath = py.import("sys")?.getattr("path")?.cast_into::<PyList>()?;
            syspath.insert(0, model.canonicalize().unwrap().parent().unwrap().display().to_string())?;
            return PyModule::import(py, model.file_name().unwrap_or(&OsStr::new("ufomodel"))).map(|m| m.unbind());
        });
        match model {
            Ok(m) => return Ok(Self { model: m }),
            Err(e) => {
                return Err(ReadError::new(
                    ReadErrorKind::PyError,
                    format!("error while importing UFO Python module:\n    {e}"),
                ));
            }
        };
    }

    fn read_particles(&self, base: &mut UFOReader) -> Result<(), ReadError> {
        match Python::attach(|py| -> PyResult<()> {
            let all_particles = self.model.bind(py).getattr("all_particles")?.cast_into::<PyList>()?;
            for p in all_particles.iter() {
                let name = extract!(p, "name")?;
                let anti_name = extract!(p, "antiname")?;
                let self_anti = name == anti_name;
                let spin = extract!(p, "spin")?;
                base.push_particle(Particle {
                    pdg_code: extract!(p, "pdg_code")?,
                    name,
                    anti_name,
                    spin,
                    color: extract!(p, "color")?,
                    mass: extract!(p, "mass" => "name")?,
                    width: extract!(p, "width" => "name")?,
                    texname: extract!(p, "texname")?,
                    anti_texname: extract!(p, "antitexname")?,
                    charge: extract!(p, "charge")?,
                    line: LineStyle::from(p.call_method0("find_line_type")?.extract::<String>()?.as_str()),
                    propagating: extract!(p, "propagating")?,
                    goldstone: extract!(p, "goldstoneboson")?,
                    propagator: extract!(p, "propagator").ok(),
                    self_anti,
                    fermi: spin % 2 == 0,
                });
            }
            Ok(())
        }) {
            Ok(()) => Ok(()),
            Err(e) => {
                return Err(ReadError::new(
                    ReadErrorKind::PyError,
                    format!("error while reading particles from UFO Python module:\n    {e}"),
                ));
            }
        }
    }

    fn read_vertices(&self, base: &mut UFOReader) -> Result<(), ReadError> {
        match Python::attach(|py| -> Result<(), ReadError> {
            let all_vertices = self.model.bind(py).getattr("all_vertices")?.cast_into::<PyList>()?;
            for v in all_vertices.iter() {
                base.push_vertex(
                    extract!(v, "name")?,
                    extract_map!(v, "particles" => "name")?,
                    extract_map!(v, "lorentz" => "name")?,
                    extract!(v, "color")?,
                    v.getattr("couplings")?
                        .cast_into::<PyDict>()?
                        .into_iter()
                        .map(|(k, v)| Ok((k.extract::<(usize, usize)>()?, v.getattr("name")?.extract::<String>()?)))
                        .collect::<Result<HashMap<_, _>, PyErr>>()?,
                );
            }
            Ok(())
        }) {
            Ok(()) => Ok(()),
            Err(e) => {
                return Err(ReadError::new(
                    ReadErrorKind::PyError,
                    format!("error while reading vertices from UFO Python module:\n    {e}"),
                ));
            }
        }
    }

    fn read_couplings(&self, base: &mut UFOReader) -> Result<(), ReadError> {
        match Python::attach(|py| -> PyResult<()> {
            let all_couplings = self.model.bind(py).getattr("all_couplings")?.cast_into::<PyList>()?;
            for c in all_couplings.iter() {
                let value = c.getattr("value")?;
                let value = if value.is_instance_of::<PyDict>() {
                    RawCouplingValue::Expansion(value.extract()?)
                } else {
                    RawCouplingValue::Finite(value.extract()?)
                };
                base.push_coupling(extract!(c, "name")?, value, extract!(c, "order")?);
            }
            Ok(())
        }) {
            Ok(()) => Ok(()),
            Err(e) => {
                return Err(ReadError::new(
                    ReadErrorKind::PyError,
                    format!("error while reading couplings from UFO Python module:\n    {e}"),
                ));
            }
        }
    }

    fn read_lorentz(&self, base: &mut UFOReader) -> Result<(), ReadError> {
        match Python::attach(|py| -> Result<(), ReadError> {
            let all_lorentz = self.model.bind(py).getattr("all_lorentz")?.cast_into::<PyList>()?;
            for l in all_lorentz.iter() {
                base.push_lorentz(extract!(l, "name")?, extract!(l, "spins")?, extract!(l, "structure")?);
            }
            Ok(())
        }) {
            Ok(()) => Ok(()),
            Err(e) => {
                return Err(ReadError::new(
                    ReadErrorKind::PyError,
                    format!("error while reading Lorentz structures from UFO Python module:\n    {e}"),
                ));
            }
        }
    }

    fn read_parameters(&self, base: &mut UFOReader) -> Result<(), ReadError> {
        match Python::attach(|py| -> PyResult<()> {
            let all_parameters = self.model.bind(py).getattr("all_parameters")?.cast_into::<PyList>()?;
            for p in all_parameters.iter() {
                let name = extract!(p, "name")?;
                let internal = p.getattr("nature")?.extract::<&str>()? == "internal";
                let value = match (internal, p.getattr("type")?.extract::<&str>()? == "real") {
                    (true, true) => RawParameterValue::InternalReal(extract!(p, "value")?),
                    (true, false) => RawParameterValue::InternalComplex(extract!(p, "value")?),
                    (false, true) => RawParameterValue::ExternalReal(extract!(p, "value")?),
                    (false, false) => RawParameterValue::ExternalComplex(extract!(p, "value")?),
                };
                base.push_parameter(
                    name,
                    value,
                    extract!(p, "texname")?,
                    if internal { None } else { Some(extract!(p, "lhablock")?) },
                    if internal { None } else { Some(extract!(p, "lhacode")?) },
                );
            }
            Ok(())
        }) {
            Ok(()) => Ok(()),
            Err(e) => {
                return Err(ReadError::new(
                    ReadErrorKind::PyError,
                    format!("error while reading parameters from UFO Python module:\n    {e}"),
                ));
            }
        }
    }

    fn read_orders(&self, base: &mut UFOReader) -> Result<(), ReadError> {
        match Python::attach(|py| -> PyResult<()> {
            let all_orders = self.model.bind(py).getattr("all_orders")?.cast_into::<PyList>()?;
            for o in all_orders.iter() {
                base.push_coupling_order(CouplingOrder {
                    name: extract!(o, "name")?,
                    expansion_order: extract!(o, "expansion_order")?,
                    hierarchy: extract!(o, "hierarchy")?,
                    perturbative_expansion: extract!(o, "perturbative_expansion")?,
                });
            }
            Ok(())
        }) {
            Ok(()) => Ok(()),
            Err(e) => {
                return Err(ReadError::new(
                    ReadErrorKind::PyError,
                    format!("error while reading coupling orders from UFO Python module:\n    {e}"),
                ));
            }
        }
    }

    fn read_ct_parameters(&self, base: &mut UFOReader) -> Result<(), ReadError> {
        match Python::attach(|py| -> PyResult<()> {
            if let Some(all_ct_parameters) = self.model.bind(py).getattr_opt("all_CTparameters")? {
                let all_ct_parameters = all_ct_parameters.cast_into::<PyList>()?;
                for p in all_ct_parameters.iter() {
                    let r#type = if p.getattr("type")?.extract::<&str>()? == "real" {
                        CTParameterType::Real
                    } else {
                        CTParameterType::Complex
                    };
                    let value = p.getattr("value")?;
                    let value = if value.is_instance_of::<PyDict>() {
                        CTParameterValue::Expansion(value.extract()?)
                    } else {
                        CTParameterValue::Finite(value.extract()?)
                    };
                    base.push_ct_parameter(CTParameter {
                        name: extract!(p, "name")?,
                        texname: extract!(p, "texname")?,
                        r#type,
                        value,
                    });
                }
            }
            Ok(())
        }) {
            Ok(()) => Ok(()),
            Err(e) => {
                return Err(ReadError::new(
                    ReadErrorKind::PyError,
                    format!("error while reading counter term parameters from UFO Python module:\n    {e}"),
                ));
            }
        }
    }

    fn read_ct_vertices(&self, base: &mut UFOReader) -> Result<(), ReadError> {
        match Python::attach(|py| -> Result<(), ReadError> {
            if let Some(all_ct_vertices) = self.model.bind(py).getattr_opt("all_CTvertices")? {
                let all_ct_vertices = all_ct_vertices.cast_into::<PyList>()?;
                for v in all_ct_vertices.iter() {
                    let name: String = extract!(v, "name")?;
                    let particles = extract_map!(v, "particles" => "name")?;
                    let lorentz = extract_map!(v, "lorentz" => "name")?;
                    let couplings = v
                        .getattr("couplings")?
                        .cast_into::<PyDict>()?
                        .into_iter()
                        .map(|(k, v)| {
                            Ok((
                                k.extract::<(usize, usize, usize)>()?,
                                v.getattr("name")?.extract::<String>()?,
                            ))
                        })
                        .collect::<Result<HashMap<_, _>, PyErr>>()?;
                    let r#type = match extract!(v, "type", &str)? {
                        "R2" => CTVertexType::R2,
                        "UV" | "UVloop" => CTVertexType::UV,
                        "UVmass" => CTVertexType::UVmass,
                        x => unreachable!("Unknown CT vertex type {x}"),
                    };
                    let loop_particles = v
                        .getattr("loop_particles")?
                        .cast_into::<PyList>()?
                        .iter()
                        .map(|l| {
                            Ok(l.cast_into::<PyList>()?
                                .iter()
                                .map(|l| {
                                    Ok(l.cast_into::<PyList>()?
                                        .iter()
                                        .map(|p| Ok(p.getattr("name")?.extract::<String>()?))
                                        .collect::<Result<_, PyErr>>()?)
                                })
                                .collect::<Result<_, PyErr>>()?)
                        })
                        .collect::<Result<_, PyErr>>()?;
                    base.push_ct_vertex(
                        name,
                        particles,
                        lorentz,
                        extract!(v, "color")?,
                        couplings,
                        r#type,
                        loop_particles,
                    );
                }
            }
            Ok(())
        }) {
            Ok(()) => Ok(()),
            Err(e) => {
                return Err(ReadError::new(
                    ReadErrorKind::PyError,
                    format!("error while reading vertices from UFO Python module:\n    {e}"),
                ));
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::{parser::UFOReader, python::cpython_reader::CPythonReader};

    #[test]
    fn cpython_ufo_import_test() {
        let path = PathBuf::from("tests/resources/UFO_QL_all_operators");
        let mut reader = UFOReader::new(&path);
        let model = reader.read_with::<CPythonReader>();
        match model {
            Err(e) => {
                reader.emit().unwrap();
                reader.emit_warnings().unwrap();
                println!("{:}", e);
                panic!()
            }
            Ok(_) => {
                reader.emit_warnings().unwrap();
            }
        }
    }

    #[test]
    fn cpython_ufo_qcd_import_test() {
        let path = PathBuf::from("tests/resources/QCD_UFO");
        let mut reader = UFOReader::new(&path);
        let model = reader.read_with::<CPythonReader>();
        match model {
            Err(e) => {
                reader.emit().unwrap();
                reader.emit_warnings().unwrap();
                println!("{:}", e);
                panic!()
            }
            Ok(m) => {
                reader.emit_warnings().unwrap();
                dbg!(m);
            }
        }
    }
}
