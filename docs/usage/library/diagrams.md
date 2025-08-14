# Generating Feynman Diagrams
Feynman diagrams can be generated in two ways in FeynGraph, by having a `DiagramGenerator` assigning particles and vertices to existing `Topology` objects or have the `DiagramGenerator` produce the diagrams directly. The second option is just a shorthand version of the first, where the `DiagramGenerator` handles the generation of the topologies internally.

## Models
At the Feynman diagram level, a full physical model is required. This is supplied via a [`Model`](../../reference/feyngraph.md#feyngraph.Model) object. FeynGraph supports fully automatic import of two commonly used model formats: [UFO 2.0](https://arxiv.org/abs/2304.09883) models through the [`from_ufo(...)`](../../reference/feyngraph.md#feyngraph.Model.from_ufo) function and [QGRAF](http://cefema-gt.tecnico.ulisboa.pt/~paulo/qgraf.html) models through the [`from_qgraf(...)`](../../reference/feyngraph.md#feyngraph.Model.from_qgraf) function.

!!! warning
    QGRAF models are supported primarily for backwards compatibility and _do not support all features_ of FeynGraph. Most notably, FeynGraph reads the analytic structure of fermionic vertices from UFO models to correctly assign signs to diagrams containing four(or more)-fermion vertices. This information is not available in QGRAF models and therefore signs of diagrams containing such vertices might not be consistent when using a QGRAF model.

The FeynGraph distribution contains only a single model, the Standard Model in Feynman gauge imported from the [default FeynRules model](https://feynrules.irmp.ucl.ac.be/wiki/StandardModel) at compile time.

## Diagram Filtering
FeynGraph generates all diagrams with a given set of incoming/outgoing external legs at the given loop order. Often only a subset of these diagrams is desired though, e.g. only diagrams with a specific power in a coupling. Constraints like these can be enforced through a [`DiagramSelector`](../../reference/feyngraph.md#feyngraph.DiagramSelector) object. Only diagrams selected by the selector are kept, all others are discarded. The `DiagramSelector` provides some predefined selection criteria for common requirements, see the [reference](../../reference/feyngraph.md#feyngraph.DiagramSelector) for a list. If these criteria are insufficient for a given requirement, the `DiagramSelector` provides the `add_custom_function` method to add a custom function to the selector. Only diagrams for which this function returns `#!python True` are kept. The same criterion can be added multiple times with different values, the `DiagramSelector` will then select diagrams satisfying any of the given values. If several different criteria are given, the diagram is required to fulfill all of them.

!!! tip
    When using the `DiagramGenerator`'s automatic topology generation, the given `DiagramSelector` is automatically converted to a `TopologySelector` and all topological selection criteria are already applied at topology level. This is not possible with custom functions, but the `DiagramSelector` provides a `add_topology_function` method to add custom functions applied at topology level through a [`TopologySelector`](topologies.md#filtering-topologies). This can significantly improve performance for complicated processes with many topologies.

!!! example
    Consider the virtual NLO QCD corrections to the electroweak production of a Higgs boson pair in the Standard Model, specifically $u\bar{u} \rightarrow HHu\bar{u}$. The Born process is at order $\mathcal{O}(\alpha^4)\mathcal{O}(\alpha_s^0)$, the NLO QCD corrections at $\mathcal{O}(\alpha^4)\mathcal{O}(\alpha_s^2)$. Additionally, some more common restrictions are applied:

    - All external legs should be on-shell, i.e. there are no self-energy insertions on an external leg.
    - Diagrams should not contain any self-loops.
    - In the current set of diagrams, there are many diagrams containing a purely electroweak loop radiating the Higgs pair. These can be interpreted as EW corrections to a QCD process with two additionally radiated Higgs bosons. To remove these, we restrict the diagrams to contain a gluon in the loop.

    === ":simple-python: Python"
        ```python
        import feyngraph as fg

        def gluon_loop(d: fg.Diagram) -> bool:
          return any(
            p.particle().name() == "g" for p in d.chord(0) # (1)!
          )

        s = fg.DiagramSelector()
        s.select_on_shell()
        s.select_self_loops(0)
        s.add_coupling_power("QED", 4)
        s.add_coupling_power("QCD", 2)
        s.add_custom_function(gluon_loop)
        ```

        1. Note that the `#!python "g"` particle name is model specific.
    === ":simple-rust: Rust"
        ```rust
        use feyngraph::prelude::*;

        fn gluon_loop(d: DiagramView<'_>) -> bool { // (1)!
          return d.chord(0).any(|p: PropagatorView<'_>| p.particle().name() == "g"); // (2)!
        }

        let mut s = DiagramSelector::new();
        s.select_on_shell();
        s.select_self_loops(0);
        s.add_coupling_power("QED", 4);
        s.add_coupling_power("QCD", 2);
        s.add_custom_function(Arc::new(gluon_loop));
        ```

        1. Note the appearance of the _View_ objects here: they provide the public interface of the underlying data objects, e.g. `DiagramView` provides the interface to the underlying `Diagram`.
        2. Note that the `#!rust "g"` particle name is model specific.

## Generating Feynman Diagrams
The final production of the Feynman diagrams is handled by the [`DiagramGenerator`](../../reference/feyngraph.md#feyngraph.DiagramGenerator). The minimal set of information needed to generate Feynman diagrams is the set of incoming/outgoing external legs, the number of loops and a model.

The `DiagramGenerator` supports two operating modes, the diagram generation mode an the topology assignment mode.

### Diagram Generation Mode
In diagram generation mode, the `DiagramGenerator`'s `generate()` function, the topologies contributing to the given process are generated automatically by the `DiagramGenerator`.

!!! example
    === ":simple-python: Python"
        ```python
        import feyngraph as fg

        diags = fg.DiagramGenerator(["u", "u~"], ["H", "H", "u", "u~"], 1, fg.Model()).generate()
        assert(len(diags) == 15966)

        def gluon_loop(d: fg.Diagram) -> bool:
          return any(
            p.particle().name() == "g" for p in d.chord(0)
          )

        s = fg.DiagramSelector()
        s.select_on_shell()
        s.select_self_loops(0)
        s.add_coupling_power("QED", 4)
        s.add_coupling_power("QCD", 2)
        s.add_custom_function(gluon_loop)
        diags = fg.DiagramGenerator(["u", "u~"], ["H", "H", "u", "u~"], 1, fg.Model(), selector = s).generate()
        assert(len(diags) == 72)
        ```
    === ":simple-rust: Rust"
        ```rust
        use feyngraph::prelude::*;

        let diags = DiagramGenerator::new(&["u", "u~"], &["H", "H", "u", "u~"], 1, Model::default(), None).generate();
        assert_eq!(diags.len(), 15966);

        fn gluon_loop(d: DiagramView<'_>) -> bool {
          return d.chord(0).any(|p: PropagatorView<'_>| p.particle().name() == "g");
        }

        let mut s = DiagramSelector::new();
        s.select_on_shell();
        s.select_self_loops(0);
        s.add_coupling_power("QED", 4);
        s.add_coupling_power("QCD", 2);
        s.add_custom_function(Arc::new(gluon_loop));

        let diags = DiagramGenerator::new(&["u", "u~"], &["H", "H", "u", "u~"], 1, Model::default(), Some(s)).generate();
        assert_eq!(diags.len(), 72);
        ```

### Assign Mode
If preprocessing on the topologies is necessary or only specific topologies are to be considered, the `DiagramGenerator` can produce Feynman diagrams from given [`Topology`](../../reference/topology.md#feyngraph.topology.Topology) objects.

!!! example
    === ":simple-python: Python"
        ```python
        import feyngraph as fg
        from feyngraph.topology import TopologyGenerator

        sm = fg.Model()
        topos = TopologyGenerator(6, 2, sm.as_topology_model()).generate()
        diags = fg.DiagramGenerator(["g", "g"], ["u", "u~", "g", "g"], 2, sm).assign_topology(topos[1833])
        assert(len(diags) == 1)
        ```
    === ":simple-rust: Rust"
        ```rust
        use feyngraph::{prelude::*, topology::TopologyGenerator};

        let sm = Model::default();
        let topos = TopologyGenerator::new(6, 2, (&sm).into(), None).generate();
        let diags = DiagramGenerator(&["g", "g"], &["u", "u~", "g", "g"], 2, sm, None).assign_topology(&topos[1833]);
        assert_eq!(diags.len(), 1);
        ```

!!! tip
    Assigning topologies with the `generate()` and `assign_topologies(...)` topologies runs fully in parallel in the Rust backend. Calling `assign_topology(...)` multiple times on different topologies runs in serial, however. It is therefore always preferable to use `generate()` or `assign_topologies(...)` if possible.

## Using Diagram Objects
When producing diagrams with a `DiagramGenerator`, it returns a `DiagramContainer` object, which is a smart container for the internal `Diagram`[^1] objects. To minimize FeynGraph's memory footprint, these internal `Diagram` objects carry as little information as possible, making them hard and unpleasant to use for further processing. For this reason, the FeynGraph Rust library provides the _View_ objects, which wrap the internal representation with a more user-friendly interface. The Python objects correspond to these view objects, not the underlying internal Rust representation.

When accessing a diagram from a `DiagramContainer`, it is automatically converted to a `DiagramView`, making the internal representation largely invisible.

[^1]: Note that this refers to the Rust `Diagram` object, _not_ the Python object.
