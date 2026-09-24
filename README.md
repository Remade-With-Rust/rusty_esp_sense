FREE RAG Converter Online -- <a href="https://RAGconverter.com">RAGconverter.com</a>

# rusty_esp_sense

[![Remade With Rust](https://img.shields.io/badge/Remade%20With-Rust-000?logo=rust&logoColor=fff)](https://github.com/remade-with-rust) [![By Mata Network](https://img.shields.io/badge/by-Mata%20Network-5b2be0)](https://www.mata.network) [![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](https://github.com/Remade-With-Rust/rusty_esp_sense/blob/main/LICENSE-MIT)

The home computer's reading of Wi-Fi channel state for the **Janus** ESP32
family. A Janus device streams every frame's raw channel state; this reads
it — live over iroh, or from a recording — and says what the room is doing,
with a model calibrated for that room. Host-only: the device keeps its
fixed-point detector, and this is what runs downstream of it.

* **It beats the chip's own detector on real data, and says by how much.**
  On 100 captures from two ESP32-C6s with every capture held out of its own
  calibration: **99.4 %** balanced against the on-chip detector's **80.0 %**
  — the gain entirely in catching the person walking (99.8 % of windows
  against 59.3 %).
* **Network traffic is not a person.** Calibrated with no traffic in the
  room and scored on captures saturated at 2–10 Mbps, it false-alarms on
  9.8 % of windows; the 2 KB linear variant on 0.1 %.
* **The number a careless benchmark would publish is in the table, labelled.**
  The raw uncentred window reads 99.8 % — and calls every empty traffic
  capture occupied, because it learned each recording session's channel
  shape rather than the person. The confound test that catches it is part
  of the benchmark.
* **A calibration is a 17 KB file.** A fixed random encoder stored as its
  seed, and a ridge readout fitted where the device is. Nothing about the
  room is guessed from somewhere else.

## What it measured

On the Universidad de Cuenca dataset (Zenodo 10.5281/zenodo.21148028, CC BY
4.0): an empty room, a person walking, an empty room saturated with UDP
traffic, and walking with traffic — 100 one-minute captures. Five folds of
whole captures; the confound test fits on the first two scenarios only and
scores traffic and a day it never saw.

| | held out | traffic false alarms, never trained on traffic | walker under traffic caught |
|---|---:|---:|---:|
| **wander + random features (default)** | **99.4 %** | 9.8 % | **98.4 %** |
| wander, linear readout (2 KB) | 96.6 % | **0.1 %** | 74.4 % |
| the on-chip detector, no training | 80.0 % | 0.3 %¹ | 61.2 %¹ |
| raw centred window + random features | 72.8 % | 39.4 % | 69.3 % |
| raw uncentred window — session leakage | 99.8 % | **100 %** | 100 % |

¹ over every capture; the detector is not trained.

**Nothing here has run against a device yet.** The dataset is someone
else's bench, with a different subcarrier layout from our own devices, and
its labels are per capture. Every number, with the run that produced it:
[`docs/LEDGER.md`](https://github.com/Remade-With-Rust/rusty_esp_sense/blob/main/docs/LEDGER.md).

## Falls and nights (W7)

**A fall is a shape**: moving, a burst above anything walking does, then
stillness that lasts. The detector is `rusty_esp_signal-core::radar::fall`
and runs on the chip; `bench-fall` is its evidence. With its thresholds set
on half the captures, the other half — 0.84 h of real walking, traffic and
empty room — raised **0 false falls**, and none until the burst threshold
was lowered from 232 ‰ to 114 ‰. On synthetic splices of real segments it
raised 10/10 falls and 0/10 "walked out of the room". **No real fall has
been recorded**, so no detection rate is claimed, and 0 events in 0.84 h
bounds false alarms only below ~3.6 an hour.

**A night is epochs**: 30 s each, empty / awake / asleep. Awake by
actigraphy's rule on motion; still epochs are asleep only when a breath
was accepted within a minute and a half, and empty otherwise — an empty
bed and a still sleeper read the same amplitude. `bench-night` plays each
Cuenca scenario back end to end: **0 of 122 empty-room epochs scored
asleep** (31 empty, 91 empty with traffic), every walking epoch awake. The
estimator accepted no breath in any of it, so the *asleep* path has not
met real data yet, and there are no sleep stages — those need a sleep
study to label.

```sh
rusty_esp_sense night recording.csv        # one JSON line per epoch, then a summary
rusty_esp_sense bench-fall <dataset-dir>
rusty_esp_sense bench-night <dataset-dir>
```

## Why not RuView's weights

RuView's released MM-Fi model takes `[3 antennas, 114 subcarriers, 10
frames]` at 100 Hz from a different radio (an ESP32 gives one antenna and
52–56 subcarriers at 50 Hz), is licensed CC BY-NC 4.0, and ships as a
pickled `.pt`. Their own study measures ~10 % cross-environment zero-shot
and finds a random frozen encoder within 2–4 points of a trained one:
channel state is distribution-locked, and the signal is in a readout fitted
where the device is. That is what this is.

## Using it

```sh
# calibrate a room from labelled recordings (csi.csv, recorded by
# rusty_esp_iroh-host's client from a device's W5 stream)
rusty_esp_sense fit --out room.safetensors empty=quiet.csv occupied=walk1.csv,walk2.csv

# read a recording: one JSON line per one-second window
rusty_esp_sense run --model room.safetensors recording.csv

# read a device live (built with --features live)
rusty_esp_sense watch --model room.safetensors <janus1 ticket>
rusty_esp_sense watch --model room.safetensors --bridge <bridge ticket>

# the benchmark
rusty_esp_sense bench-cuenca <dataset-dir>
```

```rust
use rusty_esp_sense::model::{FitConfig, Model};
use rusty_esp_sense::window::{self, WindowConfig};

let w = window::windows(&capture.samples, WindowConfig::DEFAULT);
let model = Model::fit(&w.data, &labels, vec!["empty".into(), "occupied".into()],
                       w.subcarriers, WindowConfig::DEFAULT, FitConfig::DEFAULT)?;
model.save("room.safetensors".as_ref())?;
```

**Threads.** Captures are parsed and windowed in parallel, and the
benchmark fits its folds side by side. Every command takes `--threads N`;
the default is one thread per logical CPU, or `RAYON_NUM_THREADS`. Results
do not depend on the thread count, only time and memory do.
`bench-cuenca --fit-jobs J` caps how many fits run at once, to bound peak
memory: in the raw-window configuration, 664 MB with all six at once and
264 MB one at a time.

**Calibrate with the room's normal network traffic running.** Calibrated on
four minutes without traffic, it called 13 of 60 windows of a 10 Mbps
capture occupied.

## Where it runs

On the home computer or any LAN box — not on the device. It reads what the
device's `csi-stream` output sends (raw I/Q, ~7 KB/s at 50 Hz), so a model
can be recalibrated or replaced without reflashing anything.

The whole 100-capture Cuenca benchmark runs in about 0.27 s on a 24-thread
desktop, against 3.7 s single-threaded before optimisation. Every step,
including the parallelism, is byte-identical: [docs/PERF.md](docs/PERF.md).

The tensors are [candle](https://github.com/huggingface/candle), CPU only.
`candle-core` 0.11 takes `tokenizers` with its `onig` feature, so a native
build compiles a C regex engine at build time; it is never on the model's
path, and this README does not claim "no C".

Until the Janus W5 branches merge, the sibling dependencies name those
branches, and the `live` feature builds only inside the Janus umbrella
checkout. The default build stands alone. Not yet on crates.io.

## Part of Janus

**Janus** rebuilds the Espressif ESP32 and Arduino application portfolio as
independent, memory-safe Rust packages — so a hardware maker can ship a device
that the [MATA](https://www.mata.network) home computer discovers, catalogs honestly, adopts
under its own identity, and pays for. The dependency direction never
reverses.

| layer | packages |
|---|---|
| **0 — the vocabulary** | [`rusty_esp_core`](https://crates.io/crates/rusty_esp_core) · [`rusty_esp_dsp`](https://crates.io/crates/rusty_esp_dsp) |
| **1 — the functions** | [`rusty_esp_image`](https://crates.io/crates/rusty_esp_image) · [`rusty_esp_video`](https://crates.io/crates/rusty_esp_video) · [`rusty_esp_audio`](https://crates.io/crates/rusty_esp_audio) · [`rusty_esp_signal`](https://crates.io/crates/rusty_esp_signal) · [`rusty_esp_mid`](https://crates.io/crates/rusty_esp_mid) · [`rusty_esp_iroh`](https://crates.io/crates/rusty_esp_iroh) |
| **2 — the surfaces** | [`rusty_esp_arduino`](https://crates.io/crates/rusty_esp_arduino) — the sketch facade · **`rusty_esp_sense` — the home computer's reading of the room** · `espino` — the maker's CLI (not published) |

Every package is host-verified against an external oracle and keeps a ledger
in which no number appears without the run that produced it.

Also check out the rest of [Remade With Rust](https://github.com/remade-with-rust) — including
[`rusty_alloc`](https://crates.io/crates/rusty_alloc), the pure-Rust rebuild of
mimalloc that these firmwares run on, and
[`rusty_jpeg`](https://crates.io/crates/rusty_jpeg), the JPEG engine behind the
camera path — and our sister project
[remade_ffmpeg_rs](https://github.com/Remade-With-Rust/remade_ffmpeg_rs), a ground-up Rust rebuild of FFmpeg.

## About Mata Network

[Mata Network](https://www.mata.network) builds sovereign, self-hostable infrastructure.
**Remade With Rust** is our open-source home for the permissively-licensed
building blocks that work depends on.

## License

MIT OR Apache-2.0, at your option. See [LICENSE-MIT](https://github.com/Remade-With-Rust/rusty_esp_sense/blob/main/LICENSE-MIT)
and [LICENSE-APACHE](https://github.com/Remade-With-Rust/rusty_esp_sense/blob/main/LICENSE-APACHE).
