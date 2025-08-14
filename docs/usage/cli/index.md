# Command Line Interface

For standalone usage, FeynGraph includes a command line utility called `feyngraph`. This utility is mostly intended for quick prototyping and backwards compatibility with older diagram generators, and is therefore kept relatively simple.

## Installation
The `feyngraph` command line utility is automatically installed with the FeynGraph library, it requires some additional Python packages though. They are listed in the `cli` feature and can therefore be installed with
```
pip install feyngraph[cli]
```

## Basic Usage
The `feyngraph` CLI utility requires a configuration file (see [Configuration Options](#configuration-options)) and generates a single output file from the given specification. The output is rendered through a [Jinja2 Template](https://jinja.palletsprojects.com/en/stable/), which can be either specified with the `--template` option or in the config file.
The template receives a single object, the [DiagramContainer](../../reference/feyngraph.md#feyngraph.DiagramContainer) `diags` containing all diagrams for the given specification. One template is included with FeynGraph, `json.jinja` to render the diagrams in [JSON](https://www.json.org/) format. This is also the default output format if no template is given.

## Configuration Options
`feyngraph` expects a single config file in [TOML](https://toml.io/) format. A full example configuration using all available options reads
```toml
template = "json.jinja"

[process]
incoming = ["u", "u~"]
outgoing = ["g", "g"]
loops = 2
model = "models/Standard_Model_UFO"
momenta = ["p1", "p2", "k1", "k2", "l1", "l2"]

[filter]
onshell = true
self_loops = 0
opi_components = 1
coupling_orders = { "QCD" = 2, "EW" = 0 }
custom_function = "filter"
custom_code = """
def filter(d: fg.Diagram) -> bool:
    return any(
        sum(p.id() < d.n_ext() for p in v.propagators()) >= 2 for v in d.vertices()
    )

return filter
"""

[drawing]
format = "svg"
outdir = "tmp"
filename = "d_{i}"
```
The only global option is `template`, which specifies the path to the Jinja template the output is rendered with. Note that this option is overwritten by the `--template` command line option if it is set.

### Process Specification
The physical process is specified in the `[process]` table. The possible options are

- `incoming`: List of incoming particles, specified as list of the particle names.
- `outgoing`: List of outgoing particles, specified as list of the particle names.
- `loops`: Number of loops in the generated diagrams.
- `model`[optional]: Path to the requested model. Can be the folder of a UFO model or the file of a QGRAF model. Defaults to the Standard Model in Feynman gauge if not specified.
- `momenta`[optional]: List of momentum labels to use in the output. Defaults to `p<i>` for external momenta and `l<i>` for loop momenta.

### Diagram Filters
Several options are available to restrict the set of generated diagrams, they are specified in the `[filter]` table.

- `onshell`: If `true`, keep only diagrams for which the external legs are on-shell. This is equivalent to removing all diagrams with an internal propagator carrying a single external momentum.
- `self_loops`: Keep only diagrams with the specified number of self-loops, i.e. propagators starting and ending at the same vertex.
- `coupling_orders`: Keep only diagrams with the given power in the respective coupling. Multiple couplings can be specified.

Additionally, since the `feyngraph` utility is just a small wrapper around the FeynGraph Python interface, arbitrary filters are also possible. Arbitrary Python code can be passed with the `custom_code` property, the name of the custom filter function can then be set with the `custom_function` property. This is expected to be a function taking a single [Diagram](../../reference/feyngraph.md#feyngraph.Diagram) as input and returning a `bool` as output.

!!! warning
    The custom filter option can run arbitrary code through the Python interpreter, so make sure to only use filters from trusted sources.

### Diagram Drawing
The generated diagrams can also be automatically drawn, this feature is active if the `[drawing]` table is included in the config. One output file is created for every diagram. The format can be chosen through the options

- `format`: Format of the drawing output. Supported values are `svg` and `tikz`. The default is `svg`.
- `outdir`: Directory to put the drawings in. Default is `feyngraph_drawings`.
- `filename`: Filename template for the output files. The given string is processed with the [`str.format()`](https://docs.python.org/3/library/stdtypes.html#str.format) method, which receives the single input `i`, the identifier of the diagram. The appropriate file ending is appended automatically.
