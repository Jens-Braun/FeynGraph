# Generating Topologies
!!! tip
    If no manual intervention into the topology generation is needed, the [`DiagramGenerator`](../../reference/feyngraph.md#feyngraph.DiagramGenerator) can automatically perform the topology generation step.

The first step in FeynGraph's workflow is to generate the undirected graphs possibly contributing to the physical process, called _topologies_. The topologies encapsulate the topological information and can later be assigned particles and vertices from the model to produce Feynman diagrams.

## Topology Models
Since the topologies only contain topological information, no physical model is needed at this stage. The required information is supplied via a [`TopologyModel`](../../reference/topology.md#feyngraph.topology.TopologyModel), which can be either derived from a physical model or created from a list of allowed node degrees:

=== ":simple-python: Python"
    ```python
    from feyngraph import Model
    from feyngraph.topology import TopologyModel
    sm = Model()
    assert(sm.as_topology_model(), TopologyModel([3, 4]))
    ```
=== ":simple-rust: Rust"
    ```rust
    use feyngraph::{Model, topology::TopologyModel};
    let sm = Model::default();
    assert_eq!(TopologyModel::from(&sm), TopologyModel::from(vec![3, 4]));
    ```

The allowed node degrees are the numbers of legs that can be attached to a single node, e.g. `#!python [3, 4]` in a renormalizable theory. Mixing propagators, i.e. node degree `#!python 2`, are currently not supported.

## Filtering Topologies
Often only a subset of all possible topologies is of interest, for this purpose FeynGraph provides the [`TopologySelector`](../../reference/topology.md#feyngraph.topology.TopologySelector) object to restrict the topology generation. As the name indicates, only topologies selected by the `TopologySelector` are kept, all topologies _not_ selected are filtered away. There are several predefined selection criteria, see the [API reference](../../reference/topology.md#feyngraph.topology.TopologySelector) for a list. The same criterion can be added multiple times with different values, the `TopologySelector` will then select topologies satisfying any of the given values. If several different criteria are given, the topology is required to fulfill all of them.

In addition to predefined criteria, the `TopologySelector` also supports fully custom selection functions. Custom functions can be added through the `add_custom_function` method, which takes as only input a function mapping a topology to a `#!python bool`. The topology is selected if the custom function returns `#!python True` and filtered if it returns `#!python False`.

!!! example
    Consider the 2-loop 4-point topologies in a theory with node degrees `#!python [3, 4, 5, 6]`. Say we are interested only in $s$-channel topologies containing exactly one 4-leg node and one 6-leg node. The constraint on the node degrees can be encoded as a _node partition_ criterion, which is the set of counts of the node degrees. The node partition in this example therefore is `#!python [(4, 1), (6, 1)]`. The $s$-channel constraint cannot be encoded through any of the predefined criteria, therefore a custom function is used. This function checks for nodes `#!python 0` and `#!python 1` being attached to the same node.

    === ":simple-python: Python"
        ```python
        from feyngraph.topology import TopologySelector, Topology

        def s_channel(topo: Topology) -> bool:
          return any(
            0 in node.adjacent() and 1 in node.adjacent() for node in topo.nodes()
          )

        s = TopologySelector()
        s.add_node_partition([(4, 1), (6, 1)])
        s.add_custom_function(s_channel)
        ```
    === ":simple-rust: Rust"
        ```rust
        use feyngraph::topology::{Topology, filter::TopologySelector};

        let mut s = TopologySelector::new();
        s.add_node_partition(vec![(4, 1), (6, 1)]);
        s.add_custom_function(Arc::new(
          |t| t.nodes_iter().any(|n| n.adjacent_nodes.contains(&0) && n.adjacent_nodes.contains(&1))
        ));
        ```

## Generating Topologies
The step of actually generating the topologies is handled by the [`TopologyGenerator`](../../reference/topology.md#feyngraph.topology.TopologyGenerator). The `TopologyGenerator` requires three inputs, the number of external legs, the number of loops in the topology and a `TopologyModel`. If only a subset of the topologies is to be kept, a `TopologySelector` can optionally be supplied to the `TopologyGenerator`.

!!! example

    === ":simple-python: Python"
        ```python
        from feyngraph.topology import TopologyGenerator, TopologySelector, Topology

        topo_gen = TopologyGenerator(4, 2, TopologyModel([3, 4, 5, 6]))
        topos = topo_gen.generate()
        assert(len(topos) == 2863)

        def s_channel(topo: Topology) -> bool:
            return any(
              0 in node.adjacent() and 1 in node.adjacent() for node in topo.nodes()
            )

        s = TopologySelector()
        s.add_node_partition([(4, 1), (6, 1)])
        s.add_custom_function(s_channel)

        topo_gen = TopologyGenerator(4, 2, TopologyModel([3, 4, 5, 6]), selector = s)
        topos = topo_gen.generate()
        assert(len(topos) == 9)

        ```
    === ":simple-rust: Rust"
        ```rust
        use feyngraph::topology::{TopologyGenerator, filter::TopologyFilter};

        let mut topo_gen = TopologyGenerator::new(4, 2, TopologyModel::from(vec![3, 4, 5, 6]), None);
        let topos = topo_gen.generate();
        assert_eq!(topos.len(), 2863);

        let mut s = TopologySelector::new();
        s.add_node_partition(vec![(4, 1), (6, 1)]);
        s.add_custom_function(Arc::new(
          |t| t.nodes_iter().any(|n| n.adjacent_nodes.contains(&0) && n.adjacent_nodes.contains(&1))
        ));

        let mut topo_gen = TopologyGenerator::new(4, 2, TopologyModel::from(vec![3, 4, 5, 6]), s);
        let topos = topo_gen.generate();
        assert_eq!(topos.len(), 9);
        ```

The `generate()` function returns a [`TopologContainer`](../../reference/topology.md#feyngraph.topology.TopologyContainer), which is a smart container holding the generated [`Topology`](../../reference/topology.md#feyngraph.topology.Topology) objects. It can be used like a `#!python list`/`#!rust Vec` in the respective language.
