//! A night, in thirty-second epochs: empty, awake, or asleep -- and a
//! summary a person would recognise.
//!
//! W7 of the RuView plan (`espino/docs/plans/ruview-function.md`): sleep
//! is vitals over hours, and it lives here, on the home computer, not on
//! the chip. Each epoch gets the fraction of its frames in which the on-chip
//! detector's wander says something moved, and the breathing estimator's
//! accepted rate if it had one (both from `rusty_esp_signal-core`, the same
//! code the device runs).
//!
//! # The three states, and what separates them
//!
//! - **Empty**: too little moved to be awake, and no breathing was
//!   accepted within a minute and a half. A still sleeper and an empty bed
//!   read the same amplitude; only a breath separates them. Calling an empty bed "asleep"
//!   is the failure a room sensor most needs to avoid, and it is the one
//!   `bench-night` measures on real data.
//! - **Awake / asleep**: actigraphy's rule on the motion. Cole and Kripke's
//!   relative weights over the epochs around this one (four before, two
//!   after), a weighted motion fraction, asleep below a threshold. Their
//!   weights were fitted on wrist accelerometers, not channel state; the
//!   SHAPE of the rule transfers, the threshold is a parameter, and none of
//!   it is validated against a sleep study here.
//!
//! # What is not here
//!
//! Stages. REM, light and deep sleep need polysomnography to label and to
//! validate; there is none, so there are none.

use crate::prof::{self, Counter, Stage};
use rusty_esp_signal_core::esp_core::Micros;
use rusty_esp_signal_core::radar::csi::{Config as DetectorConfig, PresenceDetector};
use rusty_esp_signal_core::radar::csi_stream::Sample;
use rusty_esp_signal_core::radar::vitals::{VitalsConfig, VitalsEstimator};

/// Cole-Kripke's relative weights for epochs t-4 … t+2.
const WEIGHTS: [f64; 7] = [106.0, 54.0, 58.0, 76.0, 230.0, 74.0, 67.0];

/// How to score.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NightConfig {
    /// Epoch length.
    pub epoch: Micros,
    /// Wander (permille) at or above which a frame counts as motion.
    pub active_permille: u16,
    /// Weighted motion fraction below which an occupied epoch is asleep.
    pub sleep_below: f64,
    /// A still epoch is asleep only when breathing was accepted within
    /// this many epochs either side; otherwise it is empty.
    pub breath_reach: usize,
    /// The recording's frame rate, for the breathing estimator (which
    /// decimates by count; see `radar::vitals`).
    pub frame_hz: u32,
}

impl NightConfig {
    /// 30 s epochs; motion at the presence threshold (32 ‰); asleep below
    /// 2 % weighted motion -- a parameter, not a validated figure; a still
    /// epoch needs a breath within three epochs (1.5 min) either side to be
    /// a sleeper; 50 Hz.
    pub const DEFAULT: NightConfig = NightConfig {
        epoch: Micros::from_secs(30),
        active_permille: 32,
        sleep_below: 0.02,
        breath_reach: 3,
        frame_hz: 50,
    };
}

impl Default for NightConfig {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// What one epoch measured.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Epoch {
    /// When it began.
    pub start: Micros,
    /// Fraction of its frames with motion.
    pub motion: f64,
    /// The last accepted breathing rate in it, tenths per minute.
    pub breathing_bpm_x10: Option<u16>,
    /// That estimate's confidence, permille.
    pub breathing_confidence: u16,
}

/// What an epoch was.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State {
    /// Nobody there.
    Empty,
    /// Somebody there, moving enough to be awake.
    Awake,
    /// Somebody there, still enough to be asleep.
    Asleep,
}

impl State {
    /// Its word.
    #[must_use]
    pub const fn word(self) -> &'static str {
        match self {
            State::Empty => "empty",
            State::Awake => "awake",
            State::Asleep => "asleep",
        }
    }
}

/// Measure a recording's epochs.
#[must_use]
pub fn epochs(samples: &[Sample], cfg: &NightConfig) -> Vec<Epoch> {
    epochs_at(samples.iter().map(|s| (s.at, s)), cfg)
}

/// [`epochs`] over samples as they come: a caller that re-times or splices
/// recordings passes them through without storing the result first.
#[must_use]
pub fn epochs_iter<I: IntoIterator<Item = Sample>>(samples: I, cfg: &NightConfig) -> Vec<Epoch> {
    epochs_at(samples.into_iter().map(|s| (s.at, s)), cfg)
}

/// [`epochs`] over `(time, sample)` pairs: each sample read at the time
/// given beside it, owned or borrowed. A caller that re-times recordings
/// (the night benchmark plays captures end to end) passes the new time and
/// a reference, instead of copying every sample to change its timestamp.
#[must_use]
pub fn epochs_at<S, I>(samples: I, cfg: &NightConfig) -> Vec<Epoch>
where
    S: core::borrow::Borrow<Sample>,
    I: IntoIterator<Item = (Micros, S)>,
{
    let _g = prof::scope(Stage::Night);
    let mut samples = samples.into_iter().peekable();
    let mut pushed = 0u64;
    let mut det = PresenceDetector::<50>::new(DetectorConfig::normalised_default());
    let mut vit: Box<VitalsEstimator<200>> =
        Box::new(VitalsEstimator::new(VitalsConfig::breathing(cfg.frame_hz)));
    let mut out = Vec::new();
    let Some(&(first, _)) = samples.peek() else {
        return out;
    };
    let mut start = first;
    let (mut frames, mut moving) = (0usize, 0usize);
    let mut breath: Option<(u16, u16)> = None;
    for (at, s) in samples {
        let s: &Sample = s.borrow();
        pushed += 1;
        while at.0 >= start.0 + cfg.epoch.0 {
            close(start, frames, moving, breath, &mut out);
            start = Micros(start.0 + cfg.epoch.0);
            frames = 0;
            moving = 0;
            breath = None;
        }
        let Ok(f) = s.features() else { continue };
        let norm = f.normalised();
        det.push(&norm, at);
        if let Some(v) = vit.push(&norm, at) {
            if v.accepted {
                breath = Some((v.bpm_x10, v.confidence));
            }
        }
        if det.warm() {
            frames += 1;
            if det.wander() >= cfg.active_permille {
                moving += 1;
            }
        }
    }
    close(start, frames, moving, breath, &mut out);
    prof::add(Counter::DetectorPushes, pushed);
    prof::add(Counter::VitalsPushes, pushed);
    prof::add(Counter::FeatureComputations, pushed);
    out
}

fn close(
    start: Micros,
    frames: usize,
    moving: usize,
    breath: Option<(u16, u16)>,
    out: &mut Vec<Epoch>,
) {
    if frames > 0 {
        out.push(Epoch {
            start,
            motion: moving as f64 / frames as f64,
            breathing_bpm_x10: breath.map(|b| b.0),
            breathing_confidence: breath.map_or(0, |b| b.1),
        });
    }
}

/// Score epochs.
///
/// Awake when the weighted motion is at or above `sleep_below`; otherwise
/// asleep when breathing was accepted within `breath_reach` epochs either
/// side, and empty when it was not. Low motion is not a sleeper by itself:
/// the first two versions of this rule let a few traffic-flickered epochs of
/// an empty room through to "asleep", because they asked the motion alone.
#[must_use]
pub fn score(epochs: &[Epoch], cfg: &NightConfig) -> Vec<State> {
    let _g = prof::scope(Stage::Night);
    let n = epochs.len();
    (0..n)
        .map(|t| {
            let mut acc = 0.0;
            let mut w = 0.0;
            for (i, &wt) in WEIGHTS.iter().enumerate() {
                let k = t as isize + i as isize - 4;
                if k >= 0 && (k as usize) < n {
                    acc += wt * epochs[k as usize].motion;
                    w += wt;
                }
            }
            let d = if w > 0.0 { acc / w } else { 0.0 };
            if d >= cfg.sleep_below {
                return State::Awake;
            }
            let lo = t.saturating_sub(cfg.breath_reach);
            let hi = (t + cfg.breath_reach).min(n.saturating_sub(1));
            if (lo..=hi).any(|k| epochs[k].breathing_bpm_x10.is_some()) {
                State::Asleep
            } else {
                State::Empty
            }
        })
        .collect()
}

/// A night, summarised.
#[derive(Debug, Clone, PartialEq)]
pub struct Summary {
    /// Epochs scored.
    pub epochs: usize,
    /// Minutes from the first occupied epoch to the last.
    pub in_room_min: f64,
    /// Minutes asleep.
    pub asleep_min: f64,
    /// Minutes from the first occupied epoch to the first of three asleep
    /// in a row; `None` if that never happened.
    pub onset_min: Option<f64>,
    /// Minutes awake after sleep began, while in the room.
    pub awake_after_onset_min: f64,
    /// Asleep ÷ in the room.
    pub efficiency: f64,
    /// Times asleep turned to awake.
    pub awakenings: usize,
    /// Median accepted breathing over asleep epochs, per minute.
    pub asleep_breathing_bpm: Option<f64>,
}

/// Summarise scored epochs.
#[must_use]
pub fn summarise(epochs: &[Epoch], states: &[State], cfg: &NightConfig) -> Summary {
    let min = cfg.epoch.0 as f64 / 60e6;
    let occupied: Vec<usize> = (0..states.len())
        .filter(|&i| states[i] != State::Empty)
        .collect();
    let (first, last) = match (occupied.first(), occupied.last()) {
        (Some(&a), Some(&b)) => (a, b),
        _ => {
            return Summary {
                epochs: states.len(),
                in_room_min: 0.0,
                asleep_min: 0.0,
                onset_min: None,
                awake_after_onset_min: 0.0,
                efficiency: 0.0,
                awakenings: 0,
                asleep_breathing_bpm: None,
            };
        }
    };
    let span = &states[first..=last];
    let asleep = span.iter().filter(|&&s| s == State::Asleep).count();
    let onset = (0..span.len().saturating_sub(2))
        .find(|&i| span[i..i + 3].iter().all(|&s| s == State::Asleep));
    let waso = onset.map_or(0, |o| {
        span[o..].iter().filter(|&&s| s == State::Awake).count()
    });
    let awakenings = span
        .windows(2)
        .filter(|w| w[0] == State::Asleep && w[1] == State::Awake)
        .count();
    let mut rates: Vec<f64> = (first..=last)
        .filter(|&i| states[i] == State::Asleep)
        .filter_map(|i| epochs[i].breathing_bpm_x10.map(|b| f64::from(b) / 10.0))
        .collect();
    rates.sort_by(f64::total_cmp);
    Summary {
        epochs: states.len(),
        in_room_min: span.len() as f64 * min,
        asleep_min: asleep as f64 * min,
        onset_min: onset.map(|o| o as f64 * min),
        awake_after_onset_min: waso as f64 * min,
        efficiency: asleep as f64 / span.len() as f64,
        awakenings,
        asleep_breathing_bpm: rates.get(rates.len() / 2).copied(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn e(motion: f64, breath: Option<u16>) -> Epoch {
        Epoch {
            start: Micros(0),
            motion,
            breathing_bpm_x10: breath,
            breathing_confidence: if breath.is_some() { 600 } else { 0 },
        }
    }

    #[test]
    fn an_empty_bed_is_empty_not_asleep() {
        let night = vec![e(0.0, None); 20];
        let s = score(&night, &NightConfig::DEFAULT);
        assert!(s.iter().all(|&x| x == State::Empty), "{s:?}");
        let sum = summarise(&night, &s, &NightConfig::DEFAULT);
        assert_eq!(sum.asleep_min, 0.0);
        assert_eq!(sum.onset_min, None);
    }

    #[test]
    fn a_still_breathing_person_is_asleep_and_a_moving_one_awake() {
        let mut night = vec![e(0.3, None); 6];
        night.extend(vec![e(0.0, Some(140)); 20]);
        night.extend(vec![e(0.25, None); 2]);
        night.extend(vec![e(0.0, Some(150)); 10]);
        let cfg = NightConfig::DEFAULT;
        let s = score(&night, &cfg);
        assert_eq!(s[0], State::Awake);
        assert_eq!(s[15], State::Asleep);
        assert!(
            !s.contains(&State::Empty),
            "breathing keeps a still room occupied"
        );
        let sum = summarise(&night, &s, &cfg);
        assert!(sum.onset_min.is_some());
        assert!(sum.awakenings >= 1, "{sum:?}");
        assert_eq!(sum.asleep_breathing_bpm, Some(14.0));
        assert!(sum.efficiency > 0.5 && sum.efficiency < 1.0, "{sum:?}");
    }

    #[test]
    fn traffic_flicker_in_an_empty_room_is_not_a_sleeper() {
        // Network traffic puts a frame or two over the motion threshold.
        let night: Vec<Epoch> = (0..12)
            .map(|i| e(if i % 4 == 0 { 0.004 } else { 0.0 }, None))
            .collect();
        let s = score(&night, &NightConfig::DEFAULT);
        assert!(s.iter().all(|&x| x == State::Empty), "{s:?}");
    }

    #[test]
    fn one_breathless_still_epoch_does_not_empty_the_room() {
        // A sleeper whose breathing the estimator missed for one epoch.
        let mut night = vec![e(0.0, Some(140)); 5];
        night.push(e(0.0, None));
        night.extend(vec![e(0.0, Some(140)); 5]);
        let s = score(&night, &NightConfig::DEFAULT);
        assert!(!s.contains(&State::Empty), "{s:?}");
    }
}
