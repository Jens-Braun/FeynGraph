"""
A modern Feynman diagram generation toolkit.
"""

from __future__ import annotations
from typing import Optional, Callable

from feyngraph.topology import Topology, TopologyModel

_WOLFRAM_ENABLED: bool


def set_threads(n_threads: int):
    """
    Set the number of threads FeynGraph will use. The default is the maximum number of available threads.

    Parameters:
        n_threads: Number of threads to use (shared across all instanced of FeynGraph running for the current process)
    """

def generate_diagrams(
        particles_in: list[str],
        particles_out: list[str],
        n_loops: int,
        model: Optional[Model] = None,
        selector: Optional[DiagramSelector] = None
        ) -> DiagramContainer :
    """
    Convenience function for diagram generation. This function only requires the minimal set of input information,
    the incoming particles and the outgoing particles. Sensible defaults are provided for all other variables.

    Examples:
    ``` python
    import feyngraph as fg
    diagrams = fg.generate_diagrams(["u", "u__tilde__"], ["u", "u__tilde"], 2)
    assert(len(diagrams), 4632)
    ```

    Parameters:
        particles_in: list of incoming particles, specified by name
        particles_out: list of outgoing particles, specified by name
        n_loops: number of loops in the generated diagrams [default: 0]
        model: model used in diagram generation [default: SM in Feynman gauge]
        selector: selector struct determining which diagrams are to be kept [default: all diagrams for zero loops, only one-particle-irreducible diagrams for loop-diagrams]

    """

class Diagram:
    """The Feynman diagram class."""

    def incoming(self) -> list[Leg]:
        """Get a list of the incoming legs."""

    def outgoing(self) -> list[Leg]:
        """Get a list of the outgoing legs."""

    def propagators(self) -> list[Propagator]:
        """Get a list of the internal propagators."""

    def propagator(self, index: int) -> Propagator:
        """Get the propagator with index `index`."""

    def vertex(self, index: int) -> Vertex:
        """Get the vertex with index `index`."""

    def vertices(self) -> list[Vertex]:
        """Get a list of the internal vertices."""

    def loop_vertices(self, index: int) -> list[Vertex]:
        """Get a list of the vertices beloning to the `index`-th loop."""

    def chord(self, index: int) -> list[Propagator]:
        """Get a list of the propagators belonging to the `index`-th loop."""

    def bridges(self) -> list[Propagator]:
        """Get a list of the bridge propagators."""

    def n_ext(self) -> int:
        """Get  the number of external legs."""

    def n_in(self) -> int:
        """Get  the number of incoming external legs."""

    def n_out(self) -> int:
        """Get  the number of outgoing external legs."""

    def symmetry_factor(self) -> int:
        """Get the diagram's symmetry factor."""

    def sign(self) -> int:
        """Get the diagram's relative sign."""

    def draw_tikz(self, file: str):
        """Draw the diagram in TikZ (TikZiT) format and write the result to `file`"""

    def draw_svg(self, file: str):
        """Draw the diagram in SVG format and write the result to `file`"""

class Leg:
    """The class representing an external leg."""

    def vertex(self, _index: int = 0) -> Vertex:
        """
        Get the vertex this leg is attached to. This function accepts an addition `_index` parameter to make its
        signature identical to `Propagator.index`, but the parameter is always ignored.
        """

    def particle(self) -> Particle:
        """Get the particle assigned to this leg."""

    def ray_index(self, _vertex: int = 0) -> int:
        """
        Get the external leg's ray index, i.e. the index of the leg of the vertex to which the external leg is
        connected to (_from the vertex perspective_). This function accepts an addition `_vertex` parameter to make its
        signature identical to `Propagator.ray_index`, but the parameter is always ignored.
        """

    def id(self) -> int:
        """Get the leg's internal id"""

    def momentum(self) -> list[int]:
        """
        Get the internal representation of the propagator's momentum. The function returns a list of integers, where
        the `i`-th entry is the coefficient of the `i`-th momentum. The first `n_ext` momenta are external, the
        remaining momenta are the `n_loops` loop momenta.
        """

    def momentum_str(self) -> str:
        """Get the string-formatted momentum flowing through the propagator."""

class Propagator:
    """The class representing an internal propagator."""

    def vertices(self) -> list[Vertex]:
        """Get a list of vertices the propagator is connected to."""

    def vertex(self, index: int) -> Vertex:
        """Get the `index`-th vertex the propagator is connected to."""

    def particle(self) -> Particle:
        """Get the particle assigned to the propagator."""

    def momentum(self) -> list[int]:
        """
        Get the internal representation of the propagator's momentum. The function returns a list of integers, where
        the `i`-th entry is the coefficient of the `i`-th momentum. The first `n_ext` momenta are external, the
        remaining momenta are the `n_loops` loop momenta.
        """

    def momentum_str(self) -> str:
        """Get the string-formatted momentum flowing through the propagator."""

    def ray_index(self, index: int) -> int:
        """
        Get the propagators ray index with respect to the `index`-th vertex it is connected to, i.e. the index of the
        leg of the `index`-th vertex to which the propagator is connected to.
        """

    def id(self) -> int:
        """Get the propagagtors internal id"""

class Vertex:
    """The class representing an internal vertex."""

    def propagators(self) -> list[Leg | Propagator]:
        """
        Get the propagators connected to this vertex. If one of the propagators is a self-loop, it will only
        appear once in the list of propagators!
        """

    def propagators_ordered(self) -> list[Leg | Propagator]:
        """
        Get the propagators connected to this vertex ordered such, that the sequence of particles matches the
        definition of the interaction in the model. If one of the propagators is a self-loop, it will only
        appear once in the list of propagators!
        """

    def interaction(self) -> InteractionVertex:
        """Get the interaction assigned to the vertex."""

    def particles_ordered(self) -> list[Particle]:
        """
        Get the particles flowing into this vertex ordered such, that the sequence of particles matches the
        definition of the interaction in the model.
        """

    def match_particles(self) -> bool:
        """Check whether the given particle names match the interaction of the vertex."""

    def id(self) -> int:
        """Get the vertex' internal id"""

    def degree(self) -> int:
        """Get the vertex' degree"""

class DiagramContainer:
    """A container of Feynman diagrams and accompanying information"""

    def query(self, selector: DiagramSelector) -> None | int:
        """
        Query whether there is a diagram in the container, which would be selected by `selector`.

        Returns:
            None if no diagram is selected, the position of the first selected diagram otherwise
        """

    def draw(self, diagrams: list[int], n_cols: Optional[int] = 4):
        """
        Draw the specified diagrams into a large canvas. Returns an SVG string, which can be displayed e.g. in a
        Jupyter notebook.

        Example:
        ```python
        from from IPython.display import SVG
        from feyngraph import generate_diagrams
        diags = generate_diagrams(["u", "u~"], ["u", "u~"], 0)
        SVG(topos.draw(range(len(diags))))
        ```

        Parameters:
            diagrams: list of IDs of diagrams to draw
            n_cols: number of topologies to draw in each row
        """

    def __len__(self) -> int:
        """"""

    def __getitem__(self, index: int) -> Diagram:
        """"""

class DiagramSelector:
    """
    A selector class which determines whether a diagram is to be kept or to be discarded. Multiple criteria can
    be specified. The available criteria are

    - opi components: select only diagrams for which the number of one-particle-irreducible components matches any of
    the given counts
    - custom functions: select only diagrams for which any of the given custom functions return `true`
    - self loops: select only diagrams which contain the specified number of self-loops
    - on-shell: select only diagrams with on-shell external legs
    - coupling powers: select only diagrams of the given power in the given coupling
    - propagator count: select only diagrams with the specified number of propagators of the given field
    - vertex count: select only diagrams with the specified number of vertices with the given fields

    For more precise definitions of each criterion, see the respective function.

    Examples:
    ```python
    selector = DiagramSelector()
    selector.select_on_shell()
    selector.select_self_loops(0)
    selector.add_coupling_power("QCD", 2)
    selector.add_coupling_power("QED", 0)
    selector.add_propagator_count("t", 0)
    ```
    """

    def select_opi_components(self, opi_count: int) :
        """
        Add a constraint to only select diagrams with `opi_count` one-particle-irreducible components.
        """

    def select_self_loops(self, count: int):
        """
        Add a constraint to only select diagrams with `count` self-loops. A self-loop is defined as an edge which ends
        on the same node it started on.
        """

    def select_on_shell(self):
        """
        Add a constraint to only select on-shell diagrams. On-shell diagrams are defined as diagrams with no self-energy
        insertions on external legs. This implementation considers internal edges carrying a single external momentum
        and no loop momentum, which is equivalent to a self-energy insertion on an external propagator.
        """

    def add_custom_function(self, py_function: Callable[[Diagram], bool]):
        """
        Add a constraint to only select diagrams for which the given function returns `true`. The function receives
        a single diagrams as input and should return a boolean.

        Examples:
        ```python
        def s_channel(diag: feyngraph.Diagram) -> bool:
            n_momenta = len(diag.propagators()[0].momentum()) # Total number of momenta in the process
            s_momentum = [1, 1]+ [0]*(n_momenta-2) # e.g. = [1, 1, 0, 0] for n_momenta = 4
            return any(propagator.momentum() == s_momentum for propagator in diag.propagators())

        selector = feyngraph.DiagramSelector()
        selector.add_custom_function(s_channel)
        ```
        """

    def add_topology_function(self, py_function: Callable[[Topology], bool]):
        """
        Add a custom topology selection function, which is used when the `DiagramSelector` is converted to a
        [`TopologySelector`](topology.md#feyngraph.topology.TopologySelector), e.g. when a
        [`DiagramGenerator`](feyngraph.md#feyngraph.DiagramGenerator) automatically generates topologies.
        """

    def select_coupling_power(self, coupling: str, power: int):
        """
        Add a constraint to only select diagrams for which the power of `coupling` sums to `power`.
        """

    def select_propagator_count(self, particle: str, count: int):
        """
        Add a constraint to only select diagrams which contain exactly `count` propagators of the field `particle`.
        """

    def select_vertex_count(self, particles: list[str], count: int):
        """
        Add a constraint to only select diagrams which contain exactly `count` vertices of the fields `particles`.
        """

    def select_vertex_degree(self, degree: int, count: int):
        """Add a criterion to only keep diagrams which contains `count` vertices of degree `degree`."""

class DiagramGenerator:
    """
    The main class used to generate Feynman diagrams.

    Examples:

    ```python
    model = Model.from_ufo("tests/Standard_Model_UFO")
    selector = DiagramSelector()
    selector.set_opi_components(1)
    diags = DiagramGenerator(["g", "g"], ["u", "u__tilde__", "g"], 1, model, selector).generate()
    assert(len(diags), 51)
    ```
    """

    def __new__(
            cls,
            incoming: list[str],
            outgoing: list[str],
            n_loops: int,
            model: Model,
            selector: DiagramSelector | None = None
            ) -> DiagramGenerator:
        """
        Create a new Diagram generator for the given process
        """

    def set_momentum_labels(self, labels: list[str]):
        """
        Set the names of the momenta. The first `n_external` ones are the external momenta, the remaining ones are
        the loop momenta. Returns an error if the number of labels does not match the diagram.
        """

    def generate(self) -> DiagramContainer:
        """
        Generate the diagrams of the given process
        """

    def count(self) -> int:
        """
        Generate the diagrams of the given process without keeping them, only retaining the total number of found
        diagrams.
        """

    def assign_topology(self, topo: Topology) -> DiagramContainer:
        """
        Assign particles and interactions to the given topology.
        """

    def assign_topologies(self, topos: list[Topology]) -> DiagramContainer:
        """
        Assign particles and interactions to the given topologies.
        """

class Model:
    """
    Internal representation of a model in FeynGraph.
    """

    def __new__(cls) -> Model:
        """Construct the default mode, the Standard Model in Feynman gauge."""

    @staticmethod
    def from_ufo(path: str) -> Model:
        """
        Import a model in the UFO format. The path should specify the folder containing the model's `.py` files.
        """

    @staticmethod
    def from_qgraf(path: str) -> Model:
        """
        Import a model in QGRAF's model format. The parser is not exhaustive in the options QGRAF supports and is only
        intended for backwards compatibility, especially for the models included in GoSam. UFO models should be
        preferred whenever possible.
        """

    def particles(self) -> list[Particle]:
        """Return the list of particles contained in the model."""

    def vertices(self) -> list[InteractionVertex]:
        """Return the list of vertices contained in the model."""

    def as_topology_model(self) -> TopologyModel:
        """Return the topology model dervied from the model."""

class Particle:
    """
    Internal representation of a particle in FeynGraph.
    """

    def name(self) -> str:
        """Get the particle's name"""

    def anti_name(self) -> str:
        """Get the name of the particle's anti particle"""

    def is_anti(self) -> bool:
        """Return true, if the particle is an anti particle (PDG ID < 0)"""

    def is_fermi(self) -> bool:
        """Return true if the particle obeys Fermi-Dirac statistics"""

    def pdg(self) -> int:
        """Get the particle's PDG ID"""

class InteractionVertex:
    """Internal representaion of an interaction vertex."""

    def coupling_orders(self):
        """Get a list of coupling orders of the interaction."""

    def name(self):
        """Get the name of the interaction vertex."""
