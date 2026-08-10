# Benchmarking

The benchmark suite measures vertical seam removal through the public
`resize()` function. Generated-image construction, file decoding, and file
output are outside the timed section. Input normalization, owned-array copies,
seam planning, and result construction are included.

Each case uses an RGB `uint8` array generated with NumPy seed `42` and the
default `GradientEnergy` method. The suite covers square images with widths of
512, 1024, and 2048 pixels. A one-seam case records the fixed-cost baseline;
the remaining cases remove 5, 50, and 200 seams to cover small, medium, and
heavy removals.
The same source array can be reused because `resize()` does not mutate it. Each
case runs one warmup round followed by five measured rounds.

This target replaces the retired stateful API. Results produced before that
change measure a different boundary and should not be compared with the new
baseline.

Results are grouped by seam count so each table compares the three image sizes
at the same removal count. Removing 200 seams from the 512-pixel image reduces
its width by about 39 percent without turning the case into a near-exhaustion
stress workload.

The current suite measures vertical removal only. Horizontal removal and mixed
vertical-horizontal resizing remain separate future benchmark additions.

The committed suite continues to measure the default backward strategy so its
reference remains comparable across releases. Forward-energy measurements are
separate because the strategy has a different cost model and is substantially
more expensive. Compare the same image sizes, seam counts, seed, warmup, rounds,
Python version, and machine when measuring both strategies. Record the forward
command, environment, medians, variability, and output-quality observations in
the development journal rather than folding an unbounded forward workload into
every routine benchmark run.

Install the development environment, then run the benchmarks explicitly:

```bash
uv sync --extra dev --frozen
uv run --frozen pytest benchmarks
```

Routine `uv run --frozen pytest` does not include benchmarks.

Run each case once without collecting timing statistics for a quicker
correctness check:

```bash
uv run --frozen pytest benchmarks --benchmark-disable
```

To measure only the smallest image size:

```bash
uv run --frozen pytest benchmarks -k 512x512
```

## Comparing changes

Save a run before changing performance-sensitive code:

```bash
uv run --frozen pytest benchmarks --benchmark-save=before
```

Run the same cases after the change and compare them with the saved result:

```bash
uv run --frozen pytest benchmarks --benchmark-compare
```

Saved results are written below `.benchmarks/`, which is ignored by Git. The
report includes the Git commit, Python runtime, machine information, repeated
measurements, and variability.

Use the same machine, power conditions, Python version, lockfile, and command
for both runs. Close unrelated CPU-intensive programs first. Results from
different environments are not directly comparable.

## v0.1.4 reference run

The current reference was measured at commit `23be2ca` (`v0.1.4`) on an Apple
M2 MacBook Air with eight cores, arm64, CPython 3.10.20, and
pytest-benchmark 5.2.3. It used NumPy seed `42`, one warmup round, and five
measured rounds. The table reports median time for each case.

| Seams | 512 × 512 | 1024 × 1024 | 2048 × 2048 |
| ---: | ---: | ---: | ---: |
| 1 | 16.61 ± 0.39 ms | 61.68 ± 0.30 ms | 238.24 ± 20.73 ms |
| 5 | 160.48 ± 50.85 ms | 250.25 ± 22.57 ms | 938.14 ± 28.23 ms |
| 50 | 607.49 ± 3.16 ms | 2.27 ± 0.08 s | 8.72 ± 0.08 s |
| 200 | 2.08 ± 0.01 s | 8.28 ± 0.03 s | 33.24 ± 0.12 s |

These numbers are a same-machine reference, not a performance guarantee. A
future comparison should report the same environment, command, inputs, and
standard deviation before drawing a conclusion.

## Quality comparisons

The test suite checks output dimensions, seam counts, single-seam connectivity,
input preservation, deterministic plans, and agreement between plan results and
previews. It does not reduce visual quality to one score.

For an algorithm change, compare the current and candidate implementations on
`examples/medium.jpg` at 400 × 240 and `examples/large.jpg` at 1000 × 900.
Generate both the resized image and the matching highlight preview, then inspect
them side by side and record visible artifacts or unwanted content loss with
the performance results. Keep comparison outputs temporary unless a case needs
to become a permanent fixture.
