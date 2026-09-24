# rusty_esp_sense

[![Remade With Rust](https://img.shields.io/badge/Remade%20With-Rust-000?logo=rust&logoColor=fff)](https://github.com/remade-with-rust) [![By Mata Network](https://img.shields.io/badge/by-Mata%20Network-5b2be0)](https://www.mata.network) [![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](https://github.com/Remade-With-Rust/rusty_esp_sense/blob/main/LICENSE-MIT)

The home computer's model over Wi-Fi channel state: windows of per-subcarrier wander, a fixed random encoder stored as its seed, and a ridge readout calibrated per room — one safetensors file. `forbid(unsafe)`, host-only, [candle](https://github.com/huggingface/candle) on the CPU.

Measured on 100 real ESP32-C6 captures with every capture held out of its own calibration: **99.4 %** balanced, against the on-chip detector's **80.0 %**, with network traffic scored separately as the false alarm it would be.

## Where the evidence is

The benchmark, every configuration tried, the session-leakage trap and the
limits are in the package's
[README](https://github.com/Remade-With-Rust/rusty_esp_sense#readme) and
[`docs/LEDGER.md`](https://github.com/Remade-With-Rust/rusty_esp_sense/blob/main/docs/LEDGER.md), where no number
appears without the run that produced it.

## Part of Janus

**Janus** rebuilds the Espressif ESP32 and Arduino application portfolio as
independent, memory-safe Rust packages — so a hardware maker can ship a device
that the [MATA](https://www.mata.network) home computer discovers, catalogs honestly, adopts
under its own identity, and pays for.

## License

MIT OR Apache-2.0, at your option.
