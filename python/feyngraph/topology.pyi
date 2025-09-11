from __future__ import annotations
from typing import Optional, Callable

class Topology:
    """The internal representation of a topology graph."""

    def edges(self) -> list[Edge]:
        """Get a list of all nodes in the topology."""

    def nodes(self) -> list[Node]:
        """Get a list of all edges in the topology."""

    def symmetry_factor(self) -> int:
        """Get the topology's symmetry factor"""

    def draw_tikz(self, path: str):
        """Draw the topology in the TikZ format"""

class Edge:
    """The internal representaion of a graph edge."""

    def nodes(self) -> list[int]:
        """Get a list of the ids of the connected nodes."""

    def momentum(self) -> list[int]:
        """
        Get the internal representation of the edge's momentum. The function returns a list of integers, where
        the `i`-th entry is the coefficient of the `i`-th momentum. The first `n_ext` momenta are external, the
        remaining momenta are the `n_loops` loop momenta.
        """

class Node:
    """The internal representation of a graph node."""

    def adjacent(self) -> list[int]:
        """Get a list of the ids of the adjacent nodes."""

    def degree(self) -> int:
        """Get the degree of the node."""

class TopologySelector:
    """
    A selector class which determines whether a topology is to be kept or to be discarded. The available critera are

    - node degrees: select only topologies for which the number of nodes with a specified degree matches any of the
    given counts
    - node partition: select only topologies matching any of the given node partitions, i.e. a topology for which the
    number of nodes of each degree exactly matches the count specified in the partition.
    - opi components: select only topologies for which the number of one-particle-irreducible components matches any of
    the given counts
    - custom functions: select only topologies for which any of the given custom functions return `true`

    """

    def select_node_degree(self, degree: int, selection: int):
        """Add a constraint to only select topologies which contain `selection` nodes of degree `degree`."""

    def select_node_degree_range(self, degree: int, start: int, end: int):
        """
        Add a constraint to only select topologies which contain between `start` and `end` nodes of degree `degree`.
        """

    def select_node_partition(self, partition: list[tuple[int, int]]):
        """
        Add a constraint to only select topologies for which the number of nodes of all given degree exactly matches
        he specified count.

        Examples:
        ```python
        selector = TopologySelector()
        # Select only topologies containing exactly four nodes of degree 3 and one node of degree 4
        selector.select_node_partition([(3, 4), (4, 1)])
        ```
        """

    def select_opi_components(self, opi_count: int):
        """Add a constraints to only select topologies with `opi_count` one-particle-irreducible components."""

    def add_custom_function(self, py_function: Callable[[Topology], bool]):
        """
        Add a constraint to only select topologies for which the given function returns `true`. The function receives
        a single topology as input and should return a boolean.

        Examples:
        ```python
        def no_self_loop(topo: feyngraph.topology.Topology) -> bool:
            return any(edge.get_nodes()[0] == edge.get_nodes()[1] for edge in topo.get_edges())

        selector = feyngraph.topology.TopologySelector()
        selector.add_custom_function(no_self_loop)
        ```
        """

    def select_on_shell(self):
        """
        Select only topologies with no self-energy insertions on external legs. This implementation considers internal
        edges carrying a single external momentum and no loop momentum, which is equivalent to a self-energy insertion
        on an external edge.
        """

    def select_self_loops(self, n: int):
        """Select only topologies containing exactly `n` self-loops."""

    def clear(self):
        """Clear all criteria."""

class TopologyModel:
    """A model containing only topological information, i.e. the allowed degrees of nodes."""

    def __new__(cls, node_degrees: list[int]) -> TopologyModel:
        """Create a new topology model containing nodes with degrees specified in `node_degrees`."""

class TopologyContainer:
    """The class representing a list of topologies."""

    def query(self, selector: TopologySelector):
        """Query whether there is a topology in the container, which would be selected by `selector`."""

    def draw(self, topologies: list[int], n_cols: Optional[int] = 4) -> str:
        """
        Draw the specified topologies into a large canvas. Returns an SVG string, which can be displayed e.g. in a
        Jupyter notebook.

        Example:
        ```python
        from from IPython.display import SVG
        from feyngraph.topology import TopologyGenerator, TopologyModel
        topos = TopologyGenerator(4, 0, TopologyModel([3, 4]))
        SVG(topos.draw(range(len(topos))))
        ```

        Parameters:
            topologies: list of IDs of topologies to draw
            n_cols: number of topologies to draw in each row
        """
    def __len__(self) -> int:
        """"""

    def __getitem__(self, index: int) -> Topology:
        """"""


class TopologyGenerator:
    """
    The main generator class of the topology module.

    Examples:
    ```python
    model = TopologyModel([3, 4])
    selector = TopologySelector()
    selector.select_opi_components(1)
    generator = TopologyGenerator(4, 3, model, selector)
    topologies = generator.generate()
    assert(len(topologies), 6166)
    ```
    """

    def __new__(cls, n_external: int, n_loops: int, model: TopologyModel, selector: Optional[TopologySelector] = None) -> TopologyGenerator:
        """
        Create a new topology generator.

        Parameters:
            selector: the selector choosing whether a given topology is kept or discarded. If no selector is specified, all topologies are kept
        """

    def generate(self) -> TopologyContainer:
        """Generate the topologies for the given configuration."""

    def count(self) -> int:
        """Generate the topologies for the given configuration without saving them, only returning the total number."""
