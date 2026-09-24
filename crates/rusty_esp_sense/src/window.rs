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

/// Cut `samples` into non-overlapping windows; a partial last window is
/// dropped.
#[must_use]
pub fn windows(samples: &[Sample], cfg: WindowConfig) -> Windows {
    let _g = prof::scope(Stage::Window);
    let t = cfg.frames.max(1);
    let mut out = Windows::default();
    let mut frames: Vec<Vec<f32>> = Vec::with_capacity(t);
    for s in samples {
        let Ok(f) = s.features() else {
            out.skipped += 1;
            continue;
        };
        let norm = f.normalised();
        let amps = norm.amplitudes();
        if out.subcarriers == 0 {
            out.subcarriers = amps.len();
        }
        if amps.len() != out.subcarriers || amps.is_empty() {
            out.skipped += 1;
            continue;
        }
        // `normalised` scales so the frame's mean is 1024.
        frames.push(amps.iter().map(|&a| f32::from(a) / 1024.0).collect());
        prof::add(Counter::FrameVecs, 1);
        prof::add(Counter::FramesWindowed, 1);
        if frames.len() == t {
            prof::add(Counter::Windows, 1);
            out.data.push(if cfg.wander {
                wander(&frames, out.subcarriers)
            } else {
                flatten(&frames, out.subcarriers, cfg.centre)
            });
            frames.clear();
        }
    }
    out
}

fn wander(frames: &[Vec<f32>], s: usize) -> Vec<f32> {
    let t = frames.len() as f32;
    (0..s)
        .map(|k| {
            let mean = frames.iter().map(|f| f[k]).sum::<f32>() / t;
            (frames.iter().map(|f| (f[k] - mean).powi(2)).sum::<f32>() / t).sqrt()
        })
        .collect()
}

fn flatten(frames: &[Vec<f32>], s: usize, centre: bool) -> Vec<f32> {
    let t = frames.len();
    let mut w = vec![0f32; s * t];
    for k in 0..s {
        let row = &mut w[k * t..(k + 1) * t];
        for (j, f) in frames.iter().enumerate() {
            row[j] = f[k];
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

#[cfg(test)]
mod tests {
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
