# Automatic Drawing of Topologies and Diagrams

!!! info
    The drawing module is in principle fully functional, but not fully tuned. Flaws like part of the drawing reaching outside of the bounding box, imperfect layout, etc. are to be expected currently.

Since the easiest method to examine Feynman diagrams is by inspecting them visually, FeynGraph provides functions for the automatic layouting and drawing of topologies and diagrams. FeynGraph currently supports two formats:

- TikZ: per-diagram output of a standalone `.tikz` file which can be directly imported in a LaTeX document via [TikZ](https://pgf-tikz.github.io/pgf/pgfmanual.pdf) or modified with [`TikZiT`](https://tikzit.github.io/). This requires the additional `feyngraph.tikzdefs` and `feyngraph.tikzstyles` files which are distributed in `<prefix>/share/FeynGraph`.
- SVG: per-diagram output of a standalone `.svg` file or a single `.svg` file with large canvas containing multiple diagrams in a grid.

Drawing functions are provided by the `Topology` and `Diagram`(`View`) objects for drawing of a single diagram and by the respective container objects for drawing of multiple diagrams.

!!! example
    === ":simple-python: Python"
        ```python
        from feyngraph import generate_diagrams
        diags = generate_diagrams(["u", "u~"], ["g", "g"], 0)
        diags.draw_svg(list(range(len(diags))))
        ```
    === ":simple-rust: Rust"
        ```rust
        use feyngraph::generate_diagrams;
        let diags = generate_diagrams(["u", "u~"], ["g", "g"], 0, None, None);
        diags.draw_svg(..diags.len());
        ```

    <figure markdown="span">
      ![Resulting Feynman Diagrams](../../assets/uubar_gg.svg){ width="900" style="background-color: white"}
      <figcaption>Resulting automatically drawn Feynman diagrams</figcaption>
    </figure>

!!! example
    === ":simple-python: Python"
        ```python
        from feyngraph.topology import TopologyGenerator, TopologySelector, TopologyModel
        s = TopologySelector()
        s.select_on_shell()
        s.select_self_loops(0)
        topos = TopologyGenerator(4, 3, TopologyModel([3, 4]), selector = s).generate()
        topos.draw([0, 1, 2])
        ```
    === ":simple-rust: Rust"
        ```rust
        use feyngraph::{model::TopologyModel, topology::{TopologyGenerator, TopologySelector}};
        let mut s = TopologySelector::new();
        s.select_on_shell();
        s.select_self_loops(0);
        let topos = TopologyGenerator::new(4, 3, TopologyModel::from(vec![3, 4]), Some(s)).generate();
        topos.draw(..3);
        ```

    <figure markdown="span">
      ![Resulting Topologies](../../assets/topos_3l.svg){ width="900" style="background-color: white"}
      <figcaption>Resulting automatically drawn Topologies</figcaption>
    </figure>

!!! tip
    The Python types implement the `_repr_svg_` method, therefore the respective objects are drawn automatically when using a Jupyter notebook. Only the first 100 topologies/diagrams of a container object are drawn with the `_repr_svg_` method.
