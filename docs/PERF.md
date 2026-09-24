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
rounding). A two-class ridge solves `y` and `−y`, and solving once and
negating was planned here. It was refuted: see R3 under Results.

# Results: forty byte-identical wins

Four passes of ten, in this order: redundancy (R), memory copies (M),
vectorisation (V), cache tiles (C). Every win left **all ten golden hashes
unchanged**. The benchmark tables, the fitted model file, `run` and
`night` are bit for bit what the analysis's baseline produced. Each win is
one commit that carries its evidence.

## Headline: the baseline commit against the last win

Method: profile builds of `ddb74f9` (the instruments, before any win) and
`df8b7d7`, with `RAYON_NUM_THREADS=1` and 6 alternating pairs each. The new
build won every pair (z 2.45).

| command | before | after | ratio |
|---|---:|---:|---:|
| `bench-cuenca` | 3,658 ms | 1,531 ms | **0.417** |
| `bench-cuenca --raw-window` | 7,334 ms | 3,331 ms | **0.451** |
| `bench-fall` | 831 ms | 559 ms | 0.666 |
| `bench-night` | 1,938 ms | 1,664 ms | 0.857 |

| command | allocations | bytes allocated | peak live |
|---|---:|---:|---:|
| `bench-cuenca` | 370,987 → 16,892 | 2.61 GB → 0.88 GB | 221 → 117 MB |
| `bench-cuenca --raw-window` | 370,988 → 10,989 | 7.49 GB → 1.35 GB | 667 → 216 MB |
| `bench-fall` | 1,495 → 1,456 | 172 → 170 MB | 48.8 → 46.5 MB |
| `bench-night` | 1,404 → 1,354 | 299 → 166 MB | 101 → 45 MB |

Most of the remaining time in `bench-fall` and `bench-night` is in
`rusty_esp_signal-core`: the presence detector, the breathing estimator
and per-frame features. That is the chip's code and outside this crate.

## Where the time is now

`bench-cuenca`, single-threaded, one run:

| stage | ms | share |
|---|---:|---:|
| Gram | 648 | 43.5 % |
| Window (about 300 ms of it is signal-core features and detector) | 311 | 20.9 % |
| Parse | 243 | 16.3 % |
| Cholesky | 121 | 8.1 % |
| Encode | 86 | 5.7 % |
| everything else | 81 | 5.5 % |

The Gram runs at about 22 G multiply-adds per second, near one core's f64
peak. It is compute-bound, and since C1 it computes only what the solve
reads. In the raw-window configuration the largest stage is the encoder's
multiply (candle's `gemm`, 51 %).

## R: redundancy

| win | what stopped being done twice | deterministic evidence |
|---|---|---|
| R1 | the encoder is generated once, not per fit | random values 408,576 → 58,368 |
| R2 | the model-size measurement fits nothing, since size depends on shape only | Gram MACs −18.7 %, Cholesky −14.3 % |
| R3 | one pass feeds the windows and the detector | feature computations −50 % |
| R4 | the fall sweep reuses the default threshold's run | fall pushes −148 k |
| R5 | the sweep walks only wander values that occur | fall pushes −50.4 % |
| R6 | a byte parser replaces the UTF-8 pass and generic `str::parse` | UTF-8 bytes 122 M → 0; field parses 38.5 M → 55,728 |
| R7 | each walking capture's active stretch is found once | scans 20 → 10 |
| R8 | streams whose peak cannot reach a threshold are skipped | fall pushes −94.5 % |
| R9 | a stream stops at its first event | fall pushes −1,764 |
| R10 | a fold's test windows are scored in one batch | tensor builds 1,185 → 72 |

**Refuted: two-class negation.** A two-output ridge solves `y` and `−y`,
and the plan was to solve once and negate. Dead features produce signed
zeros that negation does not reproduce, and the checked fallback doubled
the Cholesky. Reverted and not counted.

## M: memory copies

| win | copy removed | evidence |
|---|---|---|
| M1 | one reused frame-major buffer replaces a heap vector per frame | allocations −86.3 % |
| M2 | folds borrow their training windows instead of cloning them | −289 MB raw, allocations −54.5 % |
| M3 | stacked rows go into a buffer sized once | −934 MB raw |
| M4 | one flat score buffer replaces a vector per window | allocations −45.8 % |
| M5 | standardisation is fused into one pass | −153 MB, −542 MB raw |
| M6 | bias and ReLU run in place over the product | −145 MB |
| M7 | the Gram reads Zᵀ as a view, not a copy | −166 MB |
| M8 | column statistics stream from f32, with no f64 temporaries | peak −31 %, −46 % raw |
| M9 | nights are scored from an iterator and re-timed on the fly | peak 101 → 45 MB |
| M10 | the fall bench streams peaks instead of whole wander streams | −2.3 MB, 49 allocations |

## V: vectorisation

Evidence is the emitted-assembly census. The vectorisation is vertical
only: independent accumulators side by side, each summed in its original
order, with no reassociation and no `mul_add`.

| win | kernel | instructions per element or MAC |
|---|---|---|
| V1 | the Cholesky diagonal's dot product runs over a slice | 9.00 → 3.38 |
| V2 | the column update runs over zipped row prefixes | 12.00 → 3.75 |
| V3 | the column update handles four rows sharing each load | loads 2.00 → 1.25 per MAC |
| V4 | the forward solve handles both right-hand sides as one packed pair | 12.00 → 3.00 |
| V5 | the back solve does the same | 11.00 → 6.50 |
| V6 | `wander` accumulates every subcarrier side by side | mean 6 → 1.38; variance 9 → 2.12 |
| V7 | `Standardize::fit`'s column sums run in their own frames | 32 scalar → 11 packed; 7.50 → 4.25 per column |
| V8 | `flatten` gathers from exact frame chunks | 10 → 6, no bounds check |
| V9 | the encoder's bias and ReLU run in their own frame | the alias check and the 27-instruction fallback loop are gone |
| V10 | capture lines are found with `memchr` | Parse faster in 16/16 pairs, ratio 0.906 |

**Refuted: a stepped iterator (`step_by`).** It was the first form of V4's
back solve and of V8's gather. Both re-test exhaustion on every element.

## C: cache tiles

The analysis's sweep showed the Cholesky losing about 30 % past L2. The
other candidates were priced by stage timing in alternating pairs.

| win | layout change | evidence |
|---|---|---|
| C1 | the Gram is computed in 128-row panels over the lower triangle only | Gram 1,170 → 775 ms, 8/8 |
| C2 | the factor is blocked by 16-column panels with a packed 16×16 tile | Cholesky 279 → 216 ms, 9/10 |
| C3 | the factor stores Lᵀ in the free upper triangle, and the back solve reads rows | back solve 1.58 → 0.35 ms at d=1024; stage 12/16 |
| C4 | each row streams its prefix once against the packed panel | Cholesky 208 → 130 ms, 10/10; sweep flat |
| C5 | fit and apply share one copy of the input | InputStd 283 → 221 ms raw, 8/8 |
| C6 | windows are standardised as they are stacked | Stack + InputStd 275 → 196 ms raw, 10/10 |
| C7 | bias and ReLU are applied as the product is read (`host::with_f32`) | Encode 103 → 95 ms, 10/10 |
| C8 | the readout standardises straight into f64 | Total 1,787 → 1,715 ms, 10/12; −105 MB |
| C9 | `apply` reads in place and writes once | InputStd 25.8 → 23.1 ms, 12/12 |
| C10 | `flatten` writes each window once, with no zero-fill | probe 0.905×; stage 9/12, below resolution |

Cache sweep of the solve, in ns per multiply-add:

| d | before C2 | after C4 |
|---:|---:|---:|
| 512 | 0.175 | 0.121 |
| 1024 | 0.240 | 0.116 |
| 2048 | 0.252 | 0.111 |

The curve is flat, so the solve is no longer memory-bound.

**Refuted because they measured worse:**

- **Gram panel height.** 64, 256 and 512 rows lost to 128 in 4/6, 5/6 and
  6/6 pairs. 128 stays.
- **A contiguous Zᵀ for the Gram.** 1.43× slower, 0/8. `gemm` packs the
  column-major view it already had faster.
- **Encoder weights stored transposed.** Flat, 2/8 in the raw
  configuration. `gemm` packs either layout.
- **Column-blocking `Standardize::fit`**, 64 columns at a time in L2. 1.16×
  slower, 0/8. The full-row passes were already prefetch-friendly streams,
  and the blocks multiplied the per-row calls.
- **Encoding in 256-row blocks.** 1.35× slower, 0/8. The multiply's
  per-call cost outweighs the cache gain.
- **A separate blocked transpose for C3.** Rows 4 KiB apart alias in L1,
  making it 1.83× and 2.67× slower at d=256 and 512. Storing Lᵀ inside the
  factor replaced it.

## Method corrections found on the way

- **The assembly census must build with LTO off.** With thin LTO, a
  crate's `--emit asm` comes from the pre-link pipeline, which skips the
  loop vectoriser and unroller. V1–V5 were first counted on that output
  and then recounted. The table above has the corrected counts.
- **The census reuses a target directory.** It must delete old `.s` files
  and run `cargo clean -p` first, or it reads stale assembly.
- **The allocation census has jitter.** Even single-threaded, repeated
  runs move by up to about ±4 allocations and ±2 MB, from per-call buffers
  inside candle. A change inside that band is not a result.
- **Inlining can lose a non-aliasing proof.** Row kernels inlined into a
  caller holding two slices of one buffer stayed scalar. As
  `#[inline(never)]` functions they vectorise (V6, V7, V9).
- **Work can move between scopes.** C8 moved a conversion from unscoped
  time into RidgeStd, so the scoped sum rose while the total fell. When
  work moves between scopes, time the whole command.

# Parallelism

Seven more commits, P1 to P7, spread independent work across threads.
Each keeps results in order, so the output does not depend on the thread
count. Every commit passed the golden gate at 24, 4 and 1 threads, and a
unit test compares the benchmark report across pool sizes and `--fit-jobs`
settings.

| commit | independent unit | ordered by | evidence, 24 threads |
|---|---|---|---|
| P1 | each capture's parse | the folder walk | bench-cuenca 1,100 → 856 ms, 7/8; bench-fall 1,236 → 724 ms, 8/8 |
| P2 | each recording's windows and detector calls | recording order | 955 → 605 ms, 8/8 |
| P3 | the five folds and the confound fit | job order, confound last | 646 → 337 ms, 8/8; raw 1,110 → 640 ms, 6/6 |
| P4 | `--threads N`, `--fit-jobs J` | | both reproduce the golden output |
| P5 | the fall benchmark's per-capture detector pass | recording order | 397 → 63 ms, 8/8 |
| P6 | the night benchmark's four scenarios | scenario order | 1,632 → 821 ms, 8/8 |
| P7 | `fit`'s per-file read and windows | file order | 12 files: 99 → 53 ms, 8/8 |

In each case the first error in sequential order is the one returned, and
the per-file messages print in the same order as before.

## Scaling, before P1 and after P7

Wall time, median of 3 per point. Before P1, only candle's matrix
multiplies used threads.

| threads | cuenca before | cuenca after | raw before | raw after | fall before | fall after | night before | night after |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1,825 | 1,801 | 4,263 | 4,182 | 668 | 695 | 1,800 | 2,056 |
| 2 | 1,516 | 958 | 2,832 | 2,148 | 638 | 350 | 2,016 | 1,117 |
| 4 | 1,266 | 565 | 2,084 | 1,641 | 622 | 199 | 1,750 | 947 |
| 8 | 1,200 | 338 | 1,993 | 1,010 | 617 | 116 | 1,729 | 878 |
| 16 | 1,277 | 280 | 1,954 | 680 | 608 | 84 | 1,777 | 863 |
| 24 | 1,152 | 269 | 1,779 | 668 | 611 | 69 | 1,840 | 867 |

The one-thread column is the same code path on both sides; its spread is
the day's machine noise.

**What still limits it:**

- **The night benchmark** is capped by its largest scenario. E3 holds 45
  of the 100 captures and is one night played through stateful
  estimators. Splitting that stream would change the estimators' state and
  therefore the output.
- **bench-cuenca** flattens from 16 threads. Six fits share the pool, and
  the longest fit's sequential Cholesky and its multiplies set the floor.

## Memory

Running fits side by side holds their buffers at once. `--fit-jobs`
trades that back. `--raw-window` at 24 threads, one run each:

| fit-jobs | time | peak |
|---:|---:|---:|
| 0, all six at once | 635 ms | 664 MB |
| 3 | 696 ms | 424 MB |
| 2 | 779 ms | 356 MB |
| 1 | 1,159 ms | 264 MB |

The default configuration peaks at about 472 MB with all fits at once,
against 165 MB before P3.

## Reading the profiler under threads

Stage times add up every thread's time, so they measure CPU. Total is the
command's wall time. Under parallelism the stages can sum past Total and
the residue reads zero, and the dump's header now says so. Verdicts in this
section come from the whole command's wall time, in alternating pairs.
The allocation census is deterministic only at one thread; at 24 threads
its peak varies from run to run.

# Inside the parallel functions: ten deterministic wins

Q1 to Q10 cut work inside the functions P1 to P7 parallelised. Every one
passed the golden gate at 24, 4 and 1 threads, and each is judged on a
deterministic count from the single-thread census.

| win | where | what stopped | evidence |
|---|---|---|---|
| Q1 | `prepare` (P2) | windows and calls grown by doubling | −700 allocations |
| Q2 | `fit_on` (P3) | the training gather grown by doubling | −130 allocations, −1.5 MB |
| Q3 | `called_occupied` (P3) | test rows collected through a zero size hint | −31 allocations, −269 KB |
| Q4 | `Model::fit` (P3) | one-hot targets collected through a zero size hint | −71 allocations, −515 KB |
| Q5 | `lower_gram` (P3) | every Gram panel copied out, then copied in | −192 allocations, −28.3 MB |
| Q6 | `run` after P2 | every recording's detector calls cloned to be counted | −100 allocations |
| Q7 | the night benchmark (P6), `epochs` | every sample copied to change its timestamp | sample copies 296,035 → 0 |
| Q8 | `wander`, every window (P2, P7) | a heap vector of means per window | −5,804 allocations |
| Q9 | `prepare` (P2) | a heap vector per window | −5,904 allocations |
| Q10 | `fit` (P7) | a heap vector per window, per file | `fit` −128; 12 files 2,119 → 598 |

**bench-cuenca's single-thread allocations went from 16,921 to 3,989
(−76 %).** Its bytes allocated fell by 32.3 MB.

**Wall time is flat.** bench-cuenca against P7 read 6/10 pairs at 24
threads and 7/10 at one thread. The multiplies dominate, and these counts
were small next to them.

**Q7 first regressed and was fixed.** Its first form fed the stateful loop
through a generic `(time, sample)` iterator: `flat_map` over the captures,
`peekable`, borrowed. It removed the copies but ran 1.7 % slower (2/20
pairs, z −3.58). Moving the loop body into `EpochBuilder`, fed from plain
nested loops, kept the copies at zero and restored the time (11/20,
ratio 0.993). **A count that falls is not a win until the level above it
has been timed.**

**Census outliers.** Twice the harness's first reading after a build
moved about 20 MB in a path the change did not touch. Three reruns
reproduced the previous figure both times. Rerun any census delta a
change cannot explain before recording it.
