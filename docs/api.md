# Python API

The top-level API has two operations:

- `resize()` returns a resized image.
- `plan()` records one resize so its result and preview use the same seams.

Both operations create owned RGB `uint8` arrays and leave the source unchanged.

## Resize an image

```python
seamop.resize(
    image,
    *,
    height,
    width,
    energy=None,
    strategy=seamop.CarvingStrategy.BACKWARD,
)
```

`image` may be:

- a filesystem path (`str` or `os.PathLike`)
- a Pillow image
- an RGB `uint8` NumPy array shaped `(height, width, 3)`
- a rectangular nested list of RGB integers from 0 through 255

Filesystem and Pillow inputs are converted to RGB. NumPy arrays must already
have the required shape and dtype. Numeric inputs are not silently clipped,
scaled, or stripped of channels.

`height` and `width` must be positive integers no larger than the source
dimensions. Both may equal their source dimension. Enlargement is rejected
because seam insertion is not implemented.

The return value is a new RGB `uint8` NumPy array:

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

When both dimensions shrink, `resize()` removes vertical seams first and then
horizontal seams. The order can affect the result and is currently fixed.

`strategy` accepts `CarvingStrategy.BACKWARD` or
`CarvingStrategy.FORWARD`. Backward strategy uses `GradientEnergy` when
`energy` is omitted. Forward strategy uses pure forward energy when `energy` is
omitted and rejects an explicit energy callable.

```python
from seamop import CarvingStrategy

result = seamop.resize(
    "examples/medium.jpg",
    width=400,
    height=240,
    strategy=CarvingStrategy.FORWARD,
)
```

The command-line interface accepts strategy strings and converts them to the
enum at its boundary. The Python API expects the enum itself.

## Plan a resize

```python
seamop.plan(
    image,
    *,
    height,
    width,
    energy=None,
    strategy=seamop.CarvingStrategy.BACKWARD,
)
```

`plan()` accepts the same inputs as `resize()` and returns a `ResizePlan`.
Planning performs the seam search immediately. The stored decisions can then
produce two outputs without repeating that work:

```python
resize_plan = seamop.plan(
    "examples/medium.jpg",
    width=400,
    height=240,
)

preview = resize_plan.preview()
result = resize_plan.result()
```

`result()` returns the carved image. `preview()` returns a source-sized image
with every planned removal colored red. Pass another RGB color when needed:

```python
preview = resize_plan.preview(color=(0, 255, 0))
```

The plan also reports its array shapes:

```python
print(resize_plan.source_shape)
print(resize_plan.target_shape)
```

`ResizePlan` keeps read-only internal arrays. `result()` and `preview()` return
new writable copies, so modifying an output does not change the plan.

Create plans with `seamop.plan()` rather than calling the `ResizePlan`
constructor.

## Energy methods

The default `GradientEnergy` method for backward strategy computes
color-gradient magnitude. Two
grayscale alternatives are included:

- `SobelEnergy`
- `LaplacianEnergy`

Pass an instance through `energy`:

```python
result = seamop.resize(
    "examples/medium.jpg",
    width=400,
    height=240,
    energy=seamop.SobelEnergy(),
)
```

A custom energy may be a function, callable object, or `EnergyMethod` subclass.
It receives an RGB `uint8` array and must return a finite, real, two-dimensional
NumPy array matching the image height and width:

```python
import numpy as np
import seamop


def red_channel_energy(image: np.ndarray) -> np.ndarray:
    return image[..., 0].astype(np.float32)


result = seamop.resize(
    "examples/medium.jpg",
    width=400,
    height=240,
    energy=red_channel_energy,
)
```

`method=` remains a deprecated compatibility alias for `energy=` during the
beta migration. Passing both names is an error.

The calculator validates every returned energy map before searching for a seam.
The command-line interface intentionally limits energy selection to the three
built-in methods.

## Advanced seam calculation

`SeamCalculator` exposes vertical seam selection for algorithm experiments:

```python
import numpy as np

from seamop.calculator import SeamCalculator

image = np.zeros((4, 5, 3), dtype=np.uint8)
mask = SeamCalculator()(image, num_seams=2)

assert mask.shape == image.shape[:2]
assert mask.sum() == 2 * image.shape[0]
```

Unlike the top-level operations, `SeamCalculator` accepts only an RGB `uint8`
NumPy array. It returns a boolean mask in the source image's coordinates and
does not mutate the array.

This advanced interface remains the backward-energy calculator. Use the
top-level `resize()` or `plan()` functions for strategy selection.

`EnergyMethod` remains available for class-based implementations:

```python
from seamop.methods import EnergyMethod
```

Subclassing it is optional because any compatible callable is accepted.

## Errors

The API reports invalid inputs before seam search:

| Condition | Error |
| --- | --- |
| Unsupported image type | `TypeError` |
| Invalid image shape, dtype, channels, or values | `ValueError` |
| Non-integer dimensions, seam counts, or color channels | `TypeError` |
| Zero, negative, or enlarged target dimensions | `ValueError` |
| Invalid energy-map type or dtype | `TypeError` |
| Invalid energy-map shape or non-finite values | `ValueError` |
| Invalid strategy type | `TypeError` |
| Explicit energy with forward strategy | `ValueError` |
| Both `energy=` and `method=` supplied | `TypeError` |
| Missing or unreadable image path | `FileNotFoundError` or `ValueError` |

Operational errors do not expose a partial public result or mutate the source
input.
