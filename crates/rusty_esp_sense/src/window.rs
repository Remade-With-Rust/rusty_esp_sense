//! Windows: what the model sees.
//!
//! Each window is `frames` consecutive frames of gain-normalised amplitude,
//! subcarrier-major (`[subcarrier][frame]`), in units of the frame's mean
//! amplitude. Gain-normalised because W1b found a receiver's gain step to be
//! the whole of the amplitude detector's false presence, and dividing it out
//! costs nothing a body does. Windows do not overlap, so a benchmark that
//! splits by window never shows the model a frame it is tested on.
//!
//! `centre` subtracts each subcarrier's mean over the window, leaving only
//! how the channel CHANGED inside it. Without it, a window also carries the
//! room's static shape -- which is what lets a model learn "which recording
//! session was this" instead of "is someone there". The Cuenca benchmark
//! measured exactly that: 99.8 % held out, and every empty capture from the
//! walking day called occupied.
//!
//! `wander` replaces the window with one number per subcarrier: its
//! standard deviation over the window. It is the statistic the on-chip
//! detector thresholds (amplitude wander), given to a fitted readout
//! instead -- presence is second-order, a body makes the channel MOVE, and
//! a linear readout of centred amplitudes averages that motion away.

use crate::prof::{self, Counter, Stage};
use rusty_esp_signal_core::radar::csi::Features;
use rusty_esp_signal_core::radar::csi_stream::Sample;

/// How to cut windows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowConfig {
    /// Frames per window: 50 is one second at 50 Hz.
    pub frames: usize,
    /// Subtract each subcarrier's mean over the window.
    pub centre: bool,
    /// One value per subcarrier -- its standard deviation over the window
    /// -- instead of the window itself.
    pub wander: bool,
}

impl WindowConfig {
    /// One second at 50 Hz, as per-subcarrier wander.
    ///
    /// Changed after the first benchmark, and said so: the default was the
    /// raw centred window, which read 72.8 % balanced with captures held
    /// out and false-alarmed on 39 % of empty-with-traffic windows when
    /// trained without them. Wander reads 99.4 % and 9.8 % under the same
    /// protocol (with the default encoder; see `docs/LEDGER.md`).
    pub const DEFAULT: WindowConfig = WindowConfig {
        frames: 50,
        centre: true,
        wander: true,
    };

    /// Values per window for `subcarriers`.
    #[must_use]
    pub const fn width(&self, subcarriers: usize) -> usize {
        if self.wander {
            subcarriers
        } else {
            subcarriers * self.frames
        }
    }
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// The windows of a run of samples.
#[derive(Debug, Clone, Default)]
pub struct Windows {
    /// Subcarriers per frame (every window has the same).
    pub subcarriers: usize,
    /// Each window: `subcarriers × frames` values, or `subcarriers` with
    /// `wander`.
    pub data: Vec<Vec<f32>>,
    /// Frames skipped: a layout this crate cannot read, or a subcarrier
    /// count that differs from the run's first frame.
    pub skipped: usize,
}

/// Windows cut one frame at a time: push each frame's normalised
/// features, get a window back when one fills. [`windows`] is this over a
/// run of samples; a caller that also needs the features for something else
/// (the benchmark's detector, a live stream) computes them once and pushes
/// them here.
#[derive(Debug, Clone)]
pub struct WindowBuilder {
    cfg: WindowConfig,
    subcarriers: usize,
    /// The window being filled, frame-major (`frames × subcarriers`), one
    /// buffer reused for every window: it was a heap vector per frame.
    frames: Vec<f32>,
    /// Frames in `frames`.
    filled: usize,
    skipped: usize,
    /// `wander`'s per-subcarrier means, reused for every window: it was a
    /// heap vector per window.
    mean: Vec<f32>,
}

impl WindowBuilder {
    /// A builder for `cfg`.
    #[must_use]
    pub fn new(cfg: WindowConfig) -> Self {
        WindowBuilder {
            cfg,
            subcarriers: 0,
            frames: Vec::new(),
            filled: 0,
            skipped: 0,
            mean: Vec::new(),
        }
    }

    /// Subcarriers per frame, set by the first frame that had any.
    #[must_use]
    pub const fn subcarriers(&self) -> usize {
        self.subcarriers
    }

    /// Frames skipped so far.
    #[must_use]
    pub const fn skipped(&self) -> usize {
        self.skipped
    }

    /// One frame: its NORMALISED features, or `None` when they could not be
    /// computed. The window, when this frame completes one.
    pub fn push(&mut self, norm: Option<&Features>) -> Option<Vec<f32>> {
        let Some(norm) = norm else {
            self.skipped += 1;
            return None;
        };
        let amps = norm.amplitudes();
        if self.subcarriers == 0 {
            self.subcarriers = amps.len();
        }
        if amps.len() != self.subcarriers || amps.is_empty() {
            self.skipped += 1;
            return None;
        }
        // `normalised` scales so the frame's mean is 1024.
        let t = self.cfg.frames.max(1);
        if self.frames.capacity() == 0 {
            self.frames.reserve_exact(t * self.subcarriers);
        }
        self.frames
            .extend(amps.iter().map(|&a| f32::from(a) / 1024.0));
        self.filled += 1;
        prof::add(Counter::FramesWindowed, 1);
        if self.filled < t {
            return None;
        }
        prof::add(Counter::Windows, 1);
        let w = if self.cfg.wander {
            wander(&self.frames, t, self.subcarriers, &mut self.mean)
        } else {
            flatten(&self.frames, t, self.subcarriers, self.cfg.centre)
        };
        self.frames.clear();
        self.filled = 0;
        Some(w)
    }
}

/// Cut `samples` into non-overlapping windows; a partial last window is
/// dropped.
#[must_use]
pub fn windows(samples: &[Sample], cfg: WindowConfig) -> Windows {
    let _g = prof::scope(Stage::Window);
    prof::add(Counter::FeatureComputations, samples.len() as u64);
    let mut b = WindowBuilder::new(cfg);
    let mut data = Vec::new();
    for s in samples {
        let norm = s.features().ok().map(|f| f.normalised());
        if let Some(w) = b.push(norm.as_ref()) {
            data.push(w);
        }
    }
    Windows {
        subcarriers: b.subcarriers,
        data,
        skipped: b.skipped,
    }
}

/// Each subcarrier's standard deviation over the `t` frames of `frames`
/// (frame-major).
///
/// Vertical: every subcarrier's accumulator side by side, advanced one
/// frame at a time over contiguous rows. Each subcarrier's sums still run
/// over its frames in order, from the `-0.0` that `Iterator::sum` starts
/// floats at, so the result is bit for bit the per-subcarrier loops it
/// replaced -- which walked down the buffer with a stride of `s`.
#[inline(never)]
fn wander(frames: &[f32], t: usize, s: usize, mean: &mut Vec<f32>) -> Vec<f32> {
    let tf = t as f32;
    // The caller's scratch, reset to the same starting values each window.
    mean.clear();
    mean.resize(s, -0.0);
    for row in frames.chunks_exact(s).take(t) {
        add_row(mean, row);
    }
    for m in mean.iter_mut() {
        *m /= tf;
    }
    let mut var = vec![-0.0f32; s];
    for row in frames.chunks_exact(s).take(t) {
        add_squared_deviations(&mut var, row, mean);
    }
    for v in &mut var {
        *v = (*v / tf).sqrt();
    }
    var
}

/// `acc[k] += row[k]`. Each lane is its own accumulator: nothing is
/// reassociated. Its own frame on purpose: with `acc` and `row` arriving as
/// separate non-aliasing parameters the loop vectorises (packed adds);
/// inlined into [`wander`] the compiler lost that proof and every element
/// went scalar (census: 30 instructions per eight, all `addss`).
#[inline(never)]
fn add_row(acc: &mut [f32], row: &[f32]) {
    for (a, &r) in acc.iter_mut().zip(row) {
        *a += r;
    }
}

/// `acc[k] += (row[k] - mean[k])²`, as [`add_row`].
#[inline(never)]
fn add_squared_deviations(acc: &mut [f32], row: &[f32], mean: &[f32]) {
    for ((a, &r), &m) in acc.iter_mut().zip(row).zip(mean) {
        let dev = r - m;
        *a += dev * dev;
    }
}

/// The window subcarrier-major, from `frames` (frame-major).
#[inline(never)]
fn flatten(frames: &[f32], t: usize, s: usize, centre: bool) -> Vec<f32> {
    debug_assert_eq!(frames.len(), t * s);
    // Appended row by row, each value written once: zero-filling the
    // window first wrote all of it twice.
    let mut w = Vec::with_capacity(s * t);
    for k in 0..s {
        // Subcarrier k of every frame. Each frame is an exact chunk of `s`
        // and `k < s`, so the column read carries no check.
        let at = w.len();
        w.extend(frames.chunks_exact(s).map(|f| f[k]));
        let row = &mut w[at..];
        if centre {
            let mean = row.iter().sum::<f32>() / t as f32;
            for v in row.iter_mut() {
                *v -= mean;
            }
        }
    }
    w
}

#[cfg(test)]
mod tests {
    /// C10's probe: `flatten` as it was (the window zero-filled, then
    /// written) against the single write, at the raw benchmark's window
    /// size, over as many windows as the benchmark cuts. Timing, best of N;
    /// not a gate (the golden hashes are).
    #[test]
    #[ignore = "timing probe"]
    fn probe_flatten_single_write() {
        fn zero_filled(frames: &[f32], t: usize, s: usize, centre: bool) -> Vec<f32> {
            let mut w = vec![0f32; s * t];
            for (k, row) in (0..s).zip(w.chunks_exact_mut(t.max(1))) {
                for (v, f) in row.iter_mut().zip(frames.chunks_exact(s)) {
                    *v = f[k];
                }
                if centre {
                    let mean = row.iter().sum::<f32>() / t as f32;
                    for v in row.iter_mut() {
                        *v -= mean;
                    }
                }
            }
            w
        }
        let (t, s, windows) = (50usize, 56usize, 5904usize);
        let frames: Vec<f32> = (0..t * s).map(|i| (i % 97) as f32 / 97.0).collect();
        for centre in [false, true] {
            let (mut old, mut new) = (u128::MAX, u128::MAX);
            for _ in 0..15 {
                let clock = std::time::Instant::now();
                let mut keep = Vec::with_capacity(windows);
                for _ in 0..windows {
                    keep.push(zero_filled(std::hint::black_box(&frames), t, s, centre));
                }
                old = old.min(clock.elapsed().as_micros());
                let a = std::hint::black_box(keep);
                let clock = std::time::Instant::now();
                let mut keep = Vec::with_capacity(windows);
                for _ in 0..windows {
                    keep.push(super::flatten(std::hint::black_box(&frames), t, s, centre));
                }
                new = new.min(clock.elapsed().as_micros());
                assert_eq!(a[0], keep[0]);
            }
            println!(
                "probe centre={centre}: zero-filled {old} us, single write {new} us, ratio {:.3}",
                new as f64 / old as f64
            );
        }
    }

    use rusty_esp_signal_core::esp_core::Micros;
    use rusty_esp_signal_core::radar::csi_stream::{TAG_LLTF_20MHZ, TAG_UNKNOWN};

    use super::*;

    fn sample(k: usize, bump: i8) -> Sample {
        let mut iq = [0i8; 128];
        for e in 0..64 {
            iq[2 * e] = 20 + if e == 10 { bump } else { 0 };
        }
        Sample::from_iq(Micros(k as u64), -40, 6, TAG_LLTF_20MHZ, &iq).unwrap()
    }

    #[test]
    fn windows_do_not_overlap_and_a_partial_one_is_dropped() {
        let s: Vec<Sample> = (0..125).map(|k| sample(k, 0)).collect();
        let w = windows(
            &s,
            WindowConfig {
                frames: 50,
                centre: false,
                wander: false,
            },
        );
        assert_eq!(w.data.len(), 2);
        assert_eq!(w.subcarriers, 52);
        assert_eq!(w.data[0].len(), 52 * 50);
        assert!(
            w.data[0].iter().all(|&v| (v - 1.0).abs() < 1e-3),
            "a flat frame is 1.0 everywhere"
        );
    }

    #[test]
    fn centring_leaves_only_what_changed() {
        let s: Vec<Sample> = (0..4)
            .map(|k| sample(k, if k % 2 == 0 { 20 } else { 0 }))
            .collect();
        let w = windows(
            &s,
            WindowConfig {
                frames: 4,
                centre: true,
                wander: false,
            },
        );
        let row: f32 = w.data[0].iter().take(4).sum();
        assert!(row.abs() < 1e-5, "each subcarrier's row sums to zero");
        assert!(
            w.data[0].iter().any(|v| v.abs() > 0.1),
            "the bumped subcarrier moves"
        );
        let flat: Vec<Sample> = (0..4).map(|k| sample(k, 0)).collect();
        let still = windows(
            &flat,
            WindowConfig {
                frames: 4,
                centre: true,
                wander: false,
            },
        );
        assert!(
            still.data[0].iter().all(|v| v.abs() < 1e-6),
            "a still room centres to nothing"
        );
    }

    #[test]
    fn an_unreadable_layout_is_skipped_and_counted() {
        let mut s: Vec<Sample> = (0..3).map(|k| sample(k, 0)).collect();
        s[1].layout = TAG_UNKNOWN;
        let w = windows(
            &s,
            WindowConfig {
                frames: 2,
                centre: false,
                wander: false,
            },
        );
        assert_eq!(w.skipped, 1);
        assert_eq!(w.data.len(), 1);
    }

    /// V6's oracle: the vertical wander equals the per-subcarrier iterator
    /// sums it replaced, bit for bit, over sizes and values that include
    /// exact zeros.
    #[test]
    fn vertical_wander_matches_the_per_subcarrier_sums() {
        let reference = |frames: &[f32], t: usize, s: usize| -> Vec<f32> {
            let tf = t as f32;
            (0..s)
                .map(|k| {
                    let at = |j: usize| frames[j * s + k];
                    let mean = (0..t).map(at).sum::<f32>() / tf;
                    ((0..t).map(|j| (at(j) - mean).powi(2)).sum::<f32>() / tf).sqrt()
                })
                .collect()
        };
        for (t, s) in [(1, 1), (2, 3), (50, 56), (50, 52), (7, 64), (100, 56)] {
            let frames: Vec<f32> = (0..t * s)
                .map(|i| {
                    if i % 11 == 0 {
                        0.0
                    } else {
                        ((i * 2_654_435_761) % 4096) as f32 / 1024.0
                    }
                })
                .collect();
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(
                bits(&wander(&frames, t, s, &mut Vec::new())),
                bits(&reference(&frames, t, s)),
                "t {t} s {s}"
            );
        }
    }

    #[test]
    fn wander_is_each_subcarriers_spread_over_the_window() {
        let s: Vec<Sample> = (0..4)
            .map(|k| sample(k, if k % 2 == 0 { 20 } else { 0 }))
            .collect();
        let cfg = WindowConfig {
            frames: 4,
            centre: true,
            wander: true,
        };
        let w = windows(&s, cfg);
        assert_eq!(w.data[0].len(), 52);
        assert_eq!(cfg.width(52), 52);
        let moving = w.data[0].iter().filter(|&&v| v > 0.05).count();
        assert!(
            moving >= 1,
            "the bumped subcarrier has spread: {:?}",
            &w.data[0][..12]
        );
        let flat: Vec<Sample> = (0..4).map(|k| sample(k, 0)).collect();
        assert!(
            windows(&flat, cfg).data[0].iter().all(|v| v.abs() < 1e-6),
            "a still room has none"
        );
    }
}
