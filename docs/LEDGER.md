# rusty_esp_sense — ledger

## W6: the home computer's model, measured on real ESP32 data — 2026-09-24

### What the plan said, and why the first step did not happen

`espino/docs/plans/ruview-function.md` W6 began: *run RuView's released
Candle weights on our stream.* Their model card and study rule that out:

| | RuView's released model | a Janus device |
|---|---|---|
| input | `[3 antennas, 114 subcarriers, 10 frames]`, 100 Hz | 1 antenna, 52 (LLTF) or 56 (C6 HT20) subcarriers, 50 Hz |
| file | `pose_mmfi_best.pt` (pickle) | — |
| licence | **CC BY-NC 4.0** | espino has a commercial backend |

And their own evidence says it would not have transferred anyway:
cross-environment zero-shot ~10 %, and a random frozen encoder within 2–4
points of a trained one (§7 of their MM-Fi study). What transfers is the
method — fixed features, a readout fitted per room — so that is what was
built.

### The dataset

Universidad de Cuenca, Zenodo 10.5281/zenodo.21148028 (CC BY 4.0): two
ESP32-C6, 50 Hz injected frames, 60 s captures, C6 natural HT20 order (56
live subcarriers). Downloaded to a scratch folder, not committed.

| scenario | the room | captures | recorded |
|---|---|---:|---|
| E1 | empty, still | 15 | 2026-05-13 |
| E2 | a person walking across the line of sight | 20 | 2026-05-20 |
| E3 | empty, UDP at 2 / 5 / 10 Mbps | 45 | 2026-05-20 |
| E4 | walking + bursty traffic | 20 | 2026-06-17 |

E3 is the hard negative (does traffic look like a person?) and the day
control (empty, on the walking day). E4 rows carry a per-frame time, which
the parser reads.

### Protocol

Non-overlapping one-second windows (50 frames); five folds of **whole
captures**, round-robin within each scenario over sorted names; every capture
scored by a fit that never saw it. Plus the **confound test**: fit on every
E1 and E2 capture, score E3 and E4 — traffic it never saw, on days it never
saw. The encoder and readout defaults (1024 features, seed `"JANUS"`,
alpha 0.01) were fixed before any score existed and not changed. The window
input was changed after the first run, and every configuration is reported.

### Results (window-level; balanced = mean over scenarios)

| input → readout | model file | held out | E1 | E2 | E3 | E4 | confound: E3 false alarms | confound: E4 caught |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **wander → 1024 random features → ridge** (default) | 17,470 B | **99.4 %** | 100.0 | 99.8 | 99.9 | 97.8 | 9.8 % | **98.4 %** |
| wander → ridge | 1,966 B | 96.6 % | 96.6 | 99.7 | 99.8 | 90.2 | **0.1 %** | 74.4 % |
| wander → ridge, 2-s windows | 1,966 B | 98.4 % | 99.1 | 100.0 | 100.0 | 94.5 | 0.0 % | 78.4 % |
| on-chip detector (W1b thresholds, no training) | — | 80.0 % | 100.0 | 59.3 | 99.7 | 61.2 | — | — |
| raw centred window → 1024 RF → ridge | 39,438 B | 72.8 % | 94.3 | 53.8 | 87.1 | 56.1 | 39.4 % | 69.3 % |
| raw centred window → 4096 RF → ridge | 88,590 B | 70.2 % | 85.4 | 58.8 | 74.6 | 62.0 | 42.3 % | 71.8 % |
| raw centred window → ridge (linear) | 67,854 B | 49.5 % | 60.6 | 39.1 | 59.4 | 38.8 | 51.0 % | 53.3 % |
| raw **uncentred** window → 1024 RF → ridge | 39,438 B | 99.8 % | 100.0 | 99.6 | 100.0 | 99.7 | **100.0 %** | 100.0 % |

Captures by majority of windows, the default: 15/15, 20/20, 45/45, 20/20
held out; 45/45 and 20/20 in the confound test.

### What it says

1. **A readout calibrated on the chip's own statistic beats the chip's
   fixed thresholds** — 99.4 % vs 80.0 % — and the whole gain is catching
   the walker (99.8 % of E2 windows vs 59.3 %), not refusing traffic, which
   the chip already does. The on-chip detector's thresholds are conservative
   by design; a per-room fit is where the sensitivity comes from.
2. **Presence is second-order.** A linear readout of centred amplitudes is
   chance (49.5 %): a body makes the channel move, and motion averages out
   linearly. Random ReLU features recover part of it (72.8 %); giving the
   readout the variability directly recovers all of it.
3. **The uncentred window is the trap.** 99.8 % held out, and it calls
   **every** empty-with-traffic capture occupied when it never saw traffic:
   it learned each recording session's static channel shape. Captures a
   minute apart share a session even when folds hold out whole captures.
   This is RuView's "distribution-locked" finding, reproduced on our data,
   and the reason the confound test is part of the benchmark and not an
   afterthought.
4. **The two wander variants trade.** Random features catch the person
   under traffic (98.4 % vs 74.4 %); the linear readout refuses traffic it
   never saw (0.1 % vs 9.8 %). The default is the first; `--features 0` is
   the second, and a 2 KB model.

### The workflow, end to end

`fit` on four one-minute captures (two empty, two walking; 240 windows;
17 KB), `run` on captures it never saw: empty 60/60, walking 59/59 — and a
10 Mbps traffic capture **13/60 occupied**. Calibration without traffic
does not refuse traffic; calibrate with the room's normal traffic running.

### What is not claimed

- Anything on our own hardware: our rig is LLTF (52 subcarriers), this is
  C6 HT20 (56); a model is per room and per layout by construction.
- Accuracy against per-frame truth: the labels are per capture.
- Pose: the regression readout and torso-PCK@20 are in `metrics`, tested;
  no pose number exists until a labelled rig does.
- "No C": `candle-core` 0.11 takes `tokenizers` → `onig_sys`, compiled at
  build time on a native target.

### Gates

21 unit tests; `cargo clippy --all-targets -D warnings` with and without
`--features live`; fmt. `watch` (the live subscriber) compiles against the
W5 branches (`rusty_esp_iroh` #4, `rusty_esp_signal` #9) through the
umbrella patch; it has not run against a device.
