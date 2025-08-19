# FeynGraph

FeynGraph is a modern Feynman diagram generation toolkit aiming to be as versatile as possible while remaining pleasant to use. The library itself is written in Rust, additional language bindings are available for Python and Wolfram Mathematica.

## Installation

=== ":simple-python: Python"
    The FeynGraph Python bindings are published to PyPI and can therefore easily installed with e.g. `pip`
    ```
    pip install feyngraph
    ```
    In addition to the library interface, there is also a more classical [command line utility](usage/cli/index.md). This requires some extra dependencies, which can be installed with e.g.
    ```
    pip install feyngraph[cli]
    ```
    ### Building From Source
    The FeynGraph Python library can also easily be built from source, which requires a [Rust toolchain](https://www.rust-lang.org/tools/install) and [`maturin`](https://www.maturin.rs/). Then, in the cloned repository, a Python wheel can be compiled by running
    ```
    maturin build -r
    ```
    The library can also immediately be installed by running
    ```
    pip install .
    ```
=== ":simple-rust: Rust"
    To use FeynGraph in a Rust project, it can simply be added to the project with
    ```
    cargo add feyngraph
    ```
    or be manually added to `Cargo.toml` as
    ```toml
    [dependencies]
    feyngraph = "0.1"
    ```
=== ":simple-wolframmathematica: Mathematica"
    FeynGraph includes experimental Wolfram Mathematica bindings which use Mathematica's `ExternalEvaluate` system to call the FeynGraph Python library. To use the interface, make sure to configure a Python environment in which FeynGraph is installed as described [here](https://reference.wolfram.com/language/workflow/ConfigurePythonForExternalEvaluate.html). Then, running
    ```mathematica
    ExternalEvaluate["Python", "import feyngraph; feyngraph.import_wolfram()"] // ToExpression
    ```
    imports the FeynGraph Mathematica interface.
