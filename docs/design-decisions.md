# Design decisions and tradeoffs

## 1. Functional public operations

**Decision:** Use `resize()` for ordinary transformations and `plan()` when one
set of seam decisions must produce both a result and a preview.

The earlier mutable `SeamCarver` class was retained while these operations were
introduced, then removed during beta. It did not cache energy maps or seam
decisions, so it offered no computational advantage over the functional API.

Benefits:

- Source inputs are not mutated.
- A failed operation cannot leave a public object partially updated.
- Explicit input and output values are easier to test, retry, and pass to a
  background worker.

Tradeoff:

- Callers performing several transformations must reassign each result.

## 2. One stateful result type

**Decision:** Keep `ResizePlan` as the only stateful public image-operation
object.

A plan stores the source, carved result, and source-coordinate removal mask.
This avoids repeating seam search when a caller needs both `result()` and
`preview()`. Stored arrays are read-only, and each output method returns an
owned copy.

Tradeoff:

- A plan retains several arrays until it is released.

## 3. Compatible energy callables

**Decision:** Accept plain functions and callable objects while retaining
`EnergyMethod` for existing class-based implementations.

`SeamCalculator` validates every returned energy map before backward search. The
map must be a finite, real, two-dimensional numeric array matching the current
image.
The calculator and `EnergyMethod` are advanced interfaces, so they remain in
their defining submodules rather than the top-level package.

Tradeoff:

- Validation and normalization add one full-map pass.

The CLI exposes only built-in methods. A CLI plugin system remains deferred
until there is a demonstrated use case.

Top-level operations use `energy=` for the backward energy callable. The
deprecated `method=` alias was removed after one beta release. The advanced
`SeamCalculator.method` attribute remains a separate interface.

## 4. Vertical search with transposed height processing

**Decision:** Keep vertical search as the internal orientation for both carving
strategies. Height reduction transposes the current image and source-coordinate
map before using the corresponding vertical logic.

Orientation is local to plan construction; callers never manage numeric
direction values. Backward and forward search have separate cost contracts, but
they share planning, compaction, and source-coordinate tracking.

## 5. Recompute energy after every seam

**Decision:** Public operations recompute the energy map after each removal.

Removal changes neighboring pixels and can change their energy. Iterative
recomputation therefore matches repeated one-seam operations. A prior
width-based batching heuristic selected different pixels without making that
quality tradeoff explicit.

The planner extracts one seam per iteration. Backward operations recompute the
energy map after each removal; forward operations recompute transition costs
from the current image. Approximate batching is not part of the planner or
public contract. Future batching experiments belong in temporary benchmark
code with an explicit result contract.

## 6. Keep constants with their owners

**Decision:** Do not maintain a general constants module.

- Gradient border energy lives in the gradient implementation.
- The default highlight color lives with `ResizePlan`.
- CLI directions remain string values owned by the `remove` command.

The retired `HORIZONTAL` and `VERTICAL` integers had no meaning after the
stateful directional methods were removed. Local ownership makes each value's
scope and compatibility status clear.

## 7. CLI and library boundaries

**Decision:** Keep both an importable library and a command-line interface.

The CLI owns argument parsing, paths, saving, logging, and exit behavior. It
maps `resize`, `remove`, and `highlight` commands onto `resize()` and `plan()`.
Cyclopts derives the command-first interface and help text from typed function
signatures and docstrings. `highlight` accepts target dimensions so its preview
matches a resize request. Positional CLI dimensions use the image convention
`WIDTH HEIGHT`; internal NumPy shapes remain height then width.
When `--output` is omitted, the CLI writes a descriptively named file in the
current directory. It refuses to overwrite an existing path. The command
vocabulary remains independent from the Python API shape.

Tradeoff:

- The CLI normalizes a source image before passing an array to a functional
  operation, which may add an owned-array copy. This can be optimized later if
  profiling shows a material cost.

## 8. NumPy-first implementation

**Decision:** Use RGB `uint8` NumPy arrays as the computational representation.

Vectorized row updates, boolean masks, and source-index arrays keep the algorithm
readable while avoiding Python loops over pixels. Array shapes and dtypes remain
runtime invariants because current type annotations do not encode dimensions.

## 9. Versioning during beta

**Decision:** The earlier internal API migration did not change version
`0.5.1`.

The intended distribution had not been released. Documentation records the
removed beta interface so older local callers have a migration path.

## 10. Distribution and package identity

**Decision:** Use `seamop` for the distribution, import package, and command,
with SeamOp as the display name. Begin this identity at version `0.1.0`.

The original paper describes seam carving as an image operator. That term
continues to fit seam removal, future insertion, alternative energy policies,
and planning without tying the project to one implementation. One shared name
keeps installation, imports, and command examples consistent.

The earlier `seamcarver` identity was never published as this project's
intended distribution, and that PyPI name belongs to an unrelated project. No
compatibility package or command alias is provided.

Version `0.1.0` begins the project's release history. Earlier versions were
internal progress markers rather than releases of the intended distribution.

## 11. Separate carving strategies from energy callables

**Decision:** Expose a public `CarvingStrategy` enum separately from the
`energy=` callable.

`CarvingStrategy.BACKWARD` uses the default gradient energy when no callable is
provided. `CarvingStrategy.FORWARD` computes transition costs from the current
image and rejects an explicit energy callable. This keeps a finite algorithm
choice distinct from the open-ended custom-energy interface.

The CLI accepts strategy strings at its boundary and converts them to the enum.
No public configuration dictionary, strategy hierarchy, or backend selector is
needed.

Tradeoff:

- Forward and backward paths have separate internal search contracts.
- The top-level energy keyword is distinct from the advanced calculator's
  existing attribute name.

## Deferred work

- Seam insertion and enlargement
- User-selectable resize ordering
- A measured approximate or accelerated mode
- Raw seam and cost-table APIs
- CLI energy plugins
