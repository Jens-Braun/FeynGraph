use std::{
    error::Error,
    fmt::{Error as FmtError, Write},
    path::Path,
};

use expression::{Dictionary, Translator};
use feyngraph_amplitude::{
    AmplitudeAtom, ExternalAtom, ToAmplitude,
    wrap::{wrap_lorentz, wrap_parameter},
};
use feyngraph_core::{DiagramGenerator, DiagramSelector};
use model::Model;
use ufomodel::{
    Reader, UFOModel,
    components::{ColorAtom, LorentzAtom, ParameterAtom},
};

struct FormTranslator;
struct FormDictionary;

impl Translator<AmplitudeAtom, FormDictionary> for FormTranslator {
    type Output = String;

    fn translate(&mut self, expr: &expression::Expression<AmplitudeAtom>) -> Result<Self::Output, std::fmt::Error> {
        let mut buf = String::new();
        self.expr(&mut buf, expr)?;
        Ok(buf)
    }
}

impl Dictionary<AmplitudeAtom, String, FormTranslator> for FormDictionary {
    fn atom(buf: &mut String, atom: &AmplitudeAtom, _ctx: &mut FormTranslator) -> Result<(), std::fmt::Error> {
        match atom {
            AmplitudeAtom::Parameter(a) => write!(
                buf,
                "{}",
                match a {
                    ParameterAtom::I => "i",
                    ParameterAtom::Constant(s) | ParameterAtom::Parameter(s) => s,
                }
            ),
            AmplitudeAtom::External(a) => match a {
                ExternalAtom::Epsilon(mu, i) => write!(buf, "eps(mu{mu}, {i})"),
                ExternalAtom::EpsilonConj(mu, i) => write!(buf, "epsbar(mu{mu}, {i})"),
                ExternalAtom::U(s, i, mass) => write!(buf, "u(s{s}, i{i}, {mass})"),
                ExternalAtom::Ubar(s, i, mass) => write!(buf, "ubar(s{s}, i{i}, {mass})"),
                ExternalAtom::V(s, i, mass) => write!(buf, "v(s{s}, i{i}, {mass})"),
                ExternalAtom::Vbar(s, i, mass) => write!(buf, "vbar(s{s}, i{i}, {mass})"),
            },
            AmplitudeAtom::Color(a) => match a {
                ColorAtom::Trivial => Ok(()),
                ColorAtom::Identity(i, j) => write!(buf, "d_(i{i}, i{j})"),
                ColorAtom::T(a, i, j) => write!(buf, "T(i{i}, i{j}, a{a})"),
                ColorAtom::f(a, b, c) => write!(buf, "f(a{a}, a{b}, a{c})"),
                ColorAtom::d(a, b, c) => write!(buf, "d(a{a}, a{b}, a{c})"),
                ColorAtom::Epsilon(i, j, k) => write!(buf, "e_(i{i}, i{j}, i{k})"),
                ColorAtom::EpsilonBar(i, j, k) => write!(buf, "e_(i{i}, i{j}, i{k})"),
                ColorAtom::T6(a, i, j) => write!(buf, "t6(a{a}, a{i}, a{j})"),
                ColorAtom::K6(a, i, j) => write!(buf, "k6(a{a}, i{i}, i{j})"),
                ColorAtom::K6Bar(a, i, j) => write!(buf, "k6bar(a{a}, i{i}, i{j})"),
            },
            AmplitudeAtom::Lorentz(a) => match a {
                LorentzAtom::Trivial => Ok(()),
                LorentzAtom::Identity(i, j) => write!(buf, "Gi(s{i}, s{j})"),
                LorentzAtom::IdentityL(i, j) => write!(buf, "D(mu{i}, mu{j})"),
                LorentzAtom::Gamma(mu, i, j) => write!(buf, "G(mu{mu}, s{i}, s{j})"),
                LorentzAtom::Gamma5(i, j) => write!(buf, "G5(s{i}, s{j})"),
                LorentzAtom::ProjM(i, j) => write!(buf, "1/2*G7(s{i}, s{j})"),
                LorentzAtom::ProjP(i, j) => write!(buf, "1/2*G6(s{i}, s{j})"),
                LorentzAtom::Sigma(mu, nu, i, j) => write!(buf, "Sigma(mu{mu}, mu{nu}, s{i}, s{j})"),
                LorentzAtom::C(i, j) => write!(buf, "C(s{i}, s{j})"),
                LorentzAtom::Metric(i, j) => write!(buf, "d_(mu{i}, mu{j})"),
                LorentzAtom::P(i, j) => write!(buf, "p(mu{i}, {j})"),
                LorentzAtom::Psq(i, j) => write!(buf, "p(mu{i}, {j})*p(mu{i}, {j})"),
                LorentzAtom::Epsilon(mu, nu, rho, sigma) => write!(buf, "e_(mu{mu}, mu{nu}, mu{rho}, mu{sigma})"),
            },
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let timer = std::time::Instant::now();
    let m = Model::from_base(UFOModel::read(
        &Path::new("tests/models/Standard_Model_UFO"),
        Reader::CPython,
    )?);
    let mut s = DiagramSelector::new();
    s.select_on_shell();
    s.select_self_loops(0);
    s.select_coupling_power("QED", 0);
    let diags = DiagramGenerator::new(&["u", "u~"], &["u", "u~"], 0, m, Some(s))?.generate();

    let mut translator = FormTranslator;
    let amp = diags
        .views()
        .map(|d| -> Result<String, FmtError> {
            let n_loops = d.n_loops();
            let mut amp = d.to_amplitude();
            let mut res = translator.translate(&mut amp.numerator)?;
            for den in amp.denominators {
                write!(
                    res,
                    "*den({}, {})",
                    translator.translate(&wrap_parameter(den.mass))?,
                    translator.translate(&wrap_lorentz(den.momentum))?
                )?;
            }
            for i in 0..n_loops {
                write!(res, "*Int(l{})", i + 1)?;
            }
            Ok(res)
        })
        .collect::<Result<Vec<String>, _>>()?
        .join(" + ");
    std::fs::write(&Path::new("amp.frm"), amp)?;
    let time = timer.elapsed();
    println!(
        "Generated {} diagrams and wrote output to amplitude to `amp.frm` in {:.5} seconds.",
        diags.len(),
        (time.as_millis() as f64) * 1E-3
    );
    Ok(())
}
