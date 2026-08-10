# Architecture

`seamop` separates input/output concerns, public resize orchestration,
repeated seam planning, one-seam search, and energy calculation.

```mermaid
flowchart TD
    Client["Python caller or CLI"]
    Core["resize() / plan()"]
    Strategy["CarvingStrategy"]
    Plan["ResizePlan construction"]
    Planner["Repeated seam planner"]
    Backward["Backward search"]
    Forward["Forward search"]
    Energy["Energy callable"]
    Result["Carved or highlighted image"]

    Client --> Core
    Core --> Strategy
    Core --> Plan
    Strategy --> Plan
    Plan --> Planner
    Planner --> Backward
    Planner --> Forward
    Backward --> Energy
    Plan --> Result
    Core --> Result
```

## Components

### Public operations

`src/seamop/core.py` exposes two ordinary entry points:

- `resize()` normalizes an image, validates target dimensions, builds a plan,
  and returns an owned carved image.
- `plan()` returns a `ResizePlan` when callers need both the carved result and a
  preview based on the same seam decisions.

Neither operation mutates the caller's input.

`CarvingStrategy` selects backward or forward seam search. Backward strategy
uses the configured energy callable; forward strategy owns its transition-cost
calculation and does not accept an energy callable.

### Resize plans

`src/seamop/_plan.py` owns multi-direction resize orchestration and the
`ResizePlan` result. A plan stores independent source, result, and removal-mask
arrays. Its internal arrays are read-only; `result()` and `preview()` return
owned copies.

Width reduction runs first. Height reduction transposes the current image and
source-coordinate map, reuses vertical seam processing, then restores the
original orientation.

### Seam calculation

`src/seamop/calculator.py` validates an energy callable's output and delegates
backward repeated removal to the private planner. It returns a boolean mask in
source-image coordinates and does not mutate its input.

`src/seamop/_planner.py` owns repeated seam removal and source-coordinate
tracking. Backward planning recomputes energy after every seam; forward planning
recomputes transition costs from the current image.

`src/seamop/_search.py` contains backward and forward dynamic-programming cost
calculation and one-seam backtracking logic.

### Energy callables

`src/seamop/methods/` contains the built-in gradient, Sobel, and Laplacian
methods. Plain functions and callable objects are also accepted. The calculator
requires a finite, real, two-dimensional numeric map matching the current image
height and width.

### Input and validation boundaries

`src/seamop/_image.py` converts supported inputs into owned RGB `uint8`
arrays. `src/seamop/_validation.py` handles integer-like dimensions, seam
counts, and RGB colors.

### CLI boundary

`src/seamop/cli.py` owns command parsing, filesystem input/output, logging,
and user-facing failures. It maps commands onto the functional API:

- `resize` passes target dimensions to `resize()`.
- `remove` converts direction and count to target dimensions, then carves a plan.
- `highlight` passes target dimensions to `plan()`, then previews the pixels that
  resizing would remove.

The CLI keeps direction strings at its boundary. The Python API does not expose
numeric direction constants.

## Data flow

1. A caller supplies an image and target dimensions.
2. Input normalization creates an owned RGB `uint8` array.
3. Target validation rejects zero, negative, or enlarged dimensions.
4. Strategy selection chooses backward or forward seam search.
5. The plan builder removes width seams, followed by height seams when needed.
6. Each removal recomputes the relevant costs, finds one connected seam, and updates the
   working image and source-coordinate map.
7. `ResizePlan` stores the final image and a source-sized removal mask.
8. The caller receives an owned carved or highlighted image.

Errors propagate without exposing a partial result or mutating the source input.

## Public boundaries

The top-level public surface is:

- `resize`
- `plan` and `ResizePlan`
- `CarvingStrategy`
- the built-in energy methods
- `__version__`

`SeamCalculator` remains available from `seamop.calculator`, and
`EnergyMethod` remains available from `seamop.methods`.

The mutable `SeamCarver` compatibility class and numeric direction constants
were retired during beta. The `seamop` distribution was first published at
version `0.1.0`, with a matching Git tag and GitHub Release.

Internal seam arrays, source-coordinate maps, cost tables, planner controls, and
default implementation constants remain private.
