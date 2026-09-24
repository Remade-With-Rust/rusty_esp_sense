# rusty_esp_sense — performance analysis

Taken 2026-09-24 with the codec-analyzer method, before any optimisation.
Every later change is judged against this.

## Instruments

- **Stage profiler** (`--features profile`): wall time and call counts per
  stage from an RAII scope at the top of each stage's function; scopes never
  nest, so the residue (`Total − Σ stages`) is real. Off, every scope and
  counter compiles to nothing.
- **Work counters**: rows parsed, frames windowed, per-frame vectors,
  windows, bytes cloned and stacked, detector and vitals pushes, random
  values generated, encoder / Gram / Cholesky multiply-adds, fall pushes,
  sample copies, tensor builds. **Deterministic**: identical across runs.
- **Allocation census**: a counting allocator in the binary (profile
  builds only; the library never declares one). **Deterministic only with
  `RAYON_NUM_THREADS=1`** — with candle's thread pool the count moves by a
  couple of allocations and a few MB between runs (per-thread gemm
  buffers); single-threaded it is identical in 4/4 and 3/3 runs.
- **The gate**: `golden.sh` runs every benchmark (five `bench-cuenca`
  configurations, `bench-fall`, `bench-night`) plus `fit`, `run` and
  `night`, strips the timing line, and hashes all ten artefacts including
  the saved model file. **Null arm: the same binary twice hashes
  identically**, so the gate sees a one-bit change. Every win must leave
  all ten hashes unchanged.
- **Timer tax**: at most 6,136 scope entries per command (~0.4 ms), so
  shares are trustworthy; nothing here is per-coefficient.

Data: the Cuenca dataset (100 captures, 296,035 frames, 123 MB of CSV) at
`F:/janus-data/cuenca`, outside the temp folder that lost it once.

## Breakdowns (median of three; release)

**`bench-cuenca`, default (wander + 1024 random features): 3,052 ms**

| stage | ms | share | calls |
|---|---:|---:|---:|
| Cholesky | 730.7 | 23.9 % | 7 |
| Gram | 596.7 | 19.6 % | 7 |
| Parse | 550.3 | 18.0 % | 200 |
| InputStd (all standardisation) | 436.4 | 14.3 % | 358 |
| Detector | 349.4 | 11.4 % | 100 |
| Window | 176.2 | 5.8 % | 100 |
| Encode | 116.2 | 3.8 % | 172 |
| residue | 79.2 | 2.6 % | |

Counters: 296,035 rows = frames windowed = **per-frame vectors allocated**;
5,904 windows; 7 encoder generations (408,576 values) for **one** encoder;
33.2 G Gram and 1.27 G Cholesky multiply-adds over **7 fits** (5 folds, the
confound, and one fitted only to measure the model file's size); 374,871
allocations, 2.66 GB allocated, 269 MB peak (threaded).

**`bench-cuenca --raw-window`: 5,769 ms** — InputStd 29.3 %, Cholesky
15.2 %, Gram 11.1 %, Encode 9.5 %, Parse 9.4 %, EncoderGen **6.6 %** (the
same 2.87 M-value encoder generated 7 times: 20.1 M values), Detector
6.1 %, Stack 4.5 %, Gather 2.3 % (**354 MB of windows cloned** into folds),
residue 2.4 %. 7.5 GB allocated, 716 MB peak.

**`bench-fall`: 1,039 ms** — Parse 59.2 %, Detector 36.7 %, Fall 2.8 %
(**17.8 M fall-detector pushes** for a threshold sweep: 118 thresholds
walked one by one).

**`bench-night`: 2,241 ms** — Night 73.5 % (the presence detector and the
breathing estimator, both `rusty_esp_signal-core`), Parse 25.0 %; **296,035
samples copied** to re-time scenarios.

**`fit` (4 captures): 130 ms** — Cholesky **79.2 %**. **`run` (1 capture):
11 ms** — Parse 54 %, Window 18 %, EncoderGen 10 % (by design: the model
file stores the seed).

## Probes

**Cache sweep, Cholesky** (ns per multiply-add, best of N):

| d | working set | ns / MAC |
|---:|---:|---:|
| 64 | 32 KiB | 0.502 |
| 256 | 512 KiB | 0.511 |
| 512 | 2 MiB | 0.530 |
| **1024** | **8 MiB** | **0.694** |
| 2048 | 32 MiB | 0.929 |

Flat while in cache, climbing past it: at the production size about 30 %
of the solve is memory. **The cache-tile gate passes for the Cholesky.**

**Bounds-check ceiling, Cholesky** (throwaway, reverted): unchecked
indexing ran at **0.28–0.30 ns/MAC in cache against 0.50** — about 40 %.
Unlike the codec where the skill learned it was free, here every inner
multiply-add indexes `a[i*d+p]` element by element and carries a check the
compiler cannot prove away. Safe slices and zipped iterators remove them
without `unsafe`.

## Classification

| stage | class | lever |
|---|---|---|
| Cholesky | compute + memory (sweep), bounds-checked scalar | vectorise across independent accumulators; transposed factor; column blocks |
| Gram | compute, already SIMD (candle gemm) | do less of it: fewer fits |
| Parse | redundant work (UTF-8 validation, generic `str::parse`) and reallocation | a byte parser; exact capacity |
| InputStd | tensors rebuilt per call; two passes where one does | cache the tensors; fuse |
| Detector | redundant: features recomputed for frames already windowed | one pass feeds both |
| Window | memory: a heap vector per frame, then a transpose | one reused frame-major buffer |
| EncoderGen | redundant: one encoder generated seven times | generate once |
| Gather / Stack | memory: windows cloned, then flattened through an iterator | borrow; exact-capacity copies |
| Fall sweep | redundant: 118 thresholds where only distinct wander values change the outcome | test only those |
| Night | `rusty_esp_signal-core` (the chip's code): out of this crate's scope; the 296,035 sample copies around it are not | re-time without copying |

## Byte-identity constraints on the kernels

The outputs are floating point and the gate is bit-for-bit, so:
horizontal reductions (a dot product summed in a different order) are off
the table; what vectorises is **vertical**: independent accumulators side by
side, each summed in its original order. No `mul_add` (fusing changes
rounding). A two-class ridge solves `y` and `−y`: IEEE rounding is
symmetric, so the second solution is the first negated exactly — checked
at runtime, not assumed.
