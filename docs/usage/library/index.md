# Usage Overview
## Convenience Functions
For convenience when generating diagrams in the 'standard' setup, FeynGraph provides the [`generate_diagrams`](../../reference/feyngraph.md#feyngraph.generate_diagrams) function. For example, the 2-loop diagrams for the process $u\bar{u} \rightarrow ggg$ with the standard settings can be generated with
=== ":simple-python: Python"
    ```python
    import feyngraph as fg
    diags = fg.generate_diagrams(["u", "u~"], ["g"]*3, 2)
    ```
=== ":simple-rust: Rust"
    ```rust
    use feyngraph::generate_diagrams;
    let diags = generate_diagrams(&["u", "u~"], &["g"; 3], 2, Default::default(), Default::default());
    ```
By default, the used model is the Standard Model in Feynman gauge imported from the default [FeynRules Standard Model](https://feynrules.irmp.ucl.ac.be/wiki/StandardModel) and no diagram selection is performed.

## Lower-level Interface
If more control is needed over the generation process than the convenience functions allow, the underlying generator objects can be used manually.
The diagram generation process in FeynGraph is split into two steps,

1. Generate all topologies possibly contributing to the process (handled by a [TopologyGenerator](../../reference/topology.md#feyngraph.topology.TopologyGenerator))
2. Assign all combinations of particles and vertices to the topologies (handled by a [DiagramGenerator](../../reference/feyngraph.md#feyngraph.DiagramGenerator))

Details on the respective steps are given on the following pages.

## Global Library Configuration
FeynGraph has some global options to control the library's general environment.

### Multithreading
FeynGraph uses the [`rayon`](https://crates.io/crates/rayon) Rust crate to parallelize the diagram generation. By default, all available logical cores of the host system are used. If this is undesired, the number of used threads can be chosen manually:

=== ":simple-python: Python"
    ```python
    import feyngraph as fg
    fg.set_threads(16)
    ```
=== ":simple-rust: Rust"
    FeynGraph uses `rayon`'s default global thread pool, which can be initialized to a specific number of threads with
    ```rust
    use rayon::ThreadPoolBuilder;
    ThreadPoolBuilder::new().num_threads(16).build_global().unwrap();
    ```
    See the `rayon` [documentation](https://docs.rs/rayon/latest/rayon/struct.ThreadPoolBuilder.html) for more details.

### Logging

=== ":simple-python: Python"
    FeynGraph uses Pythons standard [`logging`](https://docs.python.org/3/library/logging.html) interface. To see the logging messages, the logger has to be configured, e.g. with a [`RichHandler`](https://rich.readthedocs.io/en/stable/logging.html) for nice formatting:
    ```python
    import logging
    from rich.logging import RichHandler
    logging.basicConfig(
      handlers=[
        format="%(message)s",
        datefmt="[%X]",
        RichHandler(show_path=False, rich_tracebacks=True),
      ])
    ```
=== ":simple-rust: Rust"
    FeynGraph uses the [`log`](https://crates.io/crates/log) crate for logging. If FeynGraph is used in an executable, it should include a [logger](https://docs.rs/log/0.4.27/log/#available-logging-implementations) to consume and show the logs produced by FeynGraph.
