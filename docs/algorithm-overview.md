# Algorithm overview

## Goal

Seam carving removes connected, low-energy pixel paths while trying to preserve
visually important content.

## Inputs and outputs

Public operations accept NumPy arrays, nested integer lists, Pillow images, and
filesystem paths. `normalize_image()` converts them into owned RGB `uint8`
arrays shaped `(H, W, 3)`.

`resize()` accepts target height and width. `plan()` accepts the same inputs and
returns a `ResizePlan` that can produce:

- an owned carved image from `result()`
- an owned source-sized preview from `preview()`

Targets may shrink or preserve each dimension. Enlargement remains deferred.

The public `CarvingStrategy` enum selects backward or forward seam search.
Backward is the default.

## Backward energy maps

`SeamCalculator` delegates pixel-energy calculation to a compatible callable.
The returned value must be a finite, real, two-dimensional NumPy array matching
the current image height and width. The calculator copies it to `float32`.

Built-in methods are:

1. `GradientEnergy`, which computes interior gradient magnitude and assigns a
   fixed border energy
2. `SobelEnergy`, which applies Sobel operators to a grayscale image
3. `LaplacianEnergy`, which computes grayscale Laplacian magnitude

These energy maps belong to the backward strategy. Forward energy does not rank
each pixel with a standalone map.

## Forward energy

Forward energy scores the new edges created when a seam is removed. For each
pixel, the dynamic-programming recurrence considers the cost of entering from
the left-up, up, or right-up predecessor. The transition costs use neighboring
pixels from the current image, so the planner recomputes them after each seam
removal.

`forward_cumulative_costs()` and `find_forward_seam()` in
`src/seamop/_search.py` implement the vertical search. The planner removes one
seam at a time through `find_forward_seams()` and preserves source coordinates
in the same way as backward planning. An explicit `energy=` callable is invalid
with this strategy.

## One-seam search

`cumulative_costs()` in `src/seamop/_search.py` creates a `float64` cost
table from the `float32` energy map. The wider accumulator prevents finite
per-pixel values from overflowing during path accumulation.

For each row, the algorithm adds the cheapest reachable predecessor:

- interior pixels consider left-up, up, and right-up
- edge pixels consider only valid neighbors

`find_seam()` starts at the cheapest endpoint in the final row and backtracks
through the same predecessor neighborhood. It returns a boolean mask containing
one pixel per row. If no finite path exists, it raises `SeamNotFoundError`.

For an image of height `H` and width `W`, energy calculation and cumulative-cost
construction are typically `O(HW)`. Backtracking is `O(H)`.

## Repeated seam planning

`find_seams()` in `src/seamop/_planner.py` owns repeated backward search:

1. Copy the image and create a flat source-index map.
2. Compute energy for the current image.
3. Find and remove one seam from the image and index map.
4. Repeat until the requested count is reached.
5. Reconstruct a boolean mask in the source image's coordinates.

Backward planning recomputes the energy map after every removal. Forward
planning uses the same source-coordinate bookkeeping but removes one seam at a
time because transition costs depend on the current image.

## Width and height reduction

`build_plan()` in `src/seamop/_plan.py` reduces width first. It applies each
seam mask to both the working image and its source-coordinate map.

To reduce height, it transposes the current image and coordinate map, reuses the
vertical seam logic, then restores the original orientation. Removed source
indices from both directions populate one source-sized mask.

The CLI maps directional commands onto this target-based model:

- vertical removal reduces the target width
- horizontal removal reduces the target height
- highlight uses the same targets but returns the plan preview

## Result invariants

- A vertical seam contains one pixel per row.
- Adjacent seam pixels differ by at most one column.
- Distinct seams in one result do not overlap.
- Forward seams use the same one-pixel-per-row and connectivity rules.
- Removing `n` vertical seams preserves height and reduces width by `n`.
- Height reduction satisfies the corresponding transposed rules.
- The source input is not mutated.
- `result()` and `preview()` use the same recorded seam decisions.
