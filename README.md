# SeamOp

[![CI](https://github.com/okinealb/seamop/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/okinealb/seamop/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

`seamop` is a Python library and command-line tool for content-aware image
resizing. It shrinks images by removing connected paths of low-energy pixels
instead of scaling every pixel or cropping a fixed region.

| Original (1428 × 968) | Resized (1000 × 900) |
| --- | --- |
| ![Original castle image](https://raw.githubusercontent.com/okinealb/seamop/main/examples/large.jpg) | ![Content-aware resized castle image](https://raw.githubusercontent.com/okinealb/seamop/main/examples/large_resized.jpg) |

The current beta supports shrinking by seam removal. Enlargement and seam
insertion are not implemented.

## Installation

Install from PyPI:

```bash
python -m pip install seamop
```

For development, use the locked uv environment:

```bash
uv sync --extra dev --frozen
```

## Command line

Resize the smaller example to 400 by 240 pixels:

```bash
seamop resize examples/medium.jpg 400 240
```

This writes `medium_resized_400x240.jpg` in the current directory. Use
`--output` to choose another path. Existing image outputs are not overwritten.

Preview the pixels that the same resize would remove:

```bash
seamop highlight examples/medium.jpg 400 240
```

Remove ten vertical seams:

```bash
seamop remove examples/medium.jpg --direction vertical --count 10
```

The default backward strategy uses gradient energy. Select pure forward energy
with `--strategy forward`; do not combine it with `--energy`.

Other commands and options are available through the built-in help:

```bash
seamop --help
seamop resize --help
seamop remove --help
seamop highlight --help
```

CLI dimensions use `WIDTH HEIGHT`.

## Python

`resize()` accepts a filesystem path, Pillow image, RGB `uint8` NumPy array, or
nested RGB integer list. It returns a new RGB `uint8` NumPy array without
mutating the input.

```python
from PIL import Image
import seamop

result = seamop.resize(
    "examples/medium.jpg",
    width=400,
    height=240,
)

Image.fromarray(result).save("medium_resized_400x240.jpg")
```

Use `plan()` when the carved result and preview must use the same seam
decisions:

```python
resize_plan = seamop.plan(
    "examples/medium.jpg",
    width=400,
    height=240,
)

preview = resize_plan.preview()
result = resize_plan.result()
```

Both output methods return independent arrays. Calling either method does not
change the plan.

The default strategy is backward seam carving with `GradientEnergy`. The
forward strategy can be selected explicitly:

```python
from seamop import CarvingStrategy

result = seamop.resize(
    "examples/medium.jpg",
    width=400,
    height=240,
    strategy=CarvingStrategy.FORWARD,
)
```

Backward carving also accepts built-in or custom energy callables through
`energy=`. Pure forward carving does not accept an energy callable.

See the [Python API guide](docs/api.md) for input rules, custom energy methods,
errors, and the advanced seam-calculation interface.

## Documentation

- [Python API](docs/api.md)
- [Algorithm overview](docs/algorithm-overview.md)
- [Architecture](docs/architecture.md)
- [Design decisions](docs/design-decisions.md)
- [Benchmarking](docs/benchmarking.md)

## Development

Run the repository checks from the project root:

```bash
uv run --frozen ruff check src tests benchmarks
uv run --frozen ruff format --check src tests benchmarks
uv run --frozen mypy
uv run --frozen pytest --cov
uv run --frozen pytest --doctest-modules src/seamop
```

Benchmarks run separately:

```bash
uv run --frozen pytest benchmarks
```

## Limitations

- Only shrinking is supported.
- Width is reduced before height when both dimensions change.
- Results depend on the image and selected carving strategy and energy method.
- Large reductions can distort important content.

## License

[MIT](LICENSE)
