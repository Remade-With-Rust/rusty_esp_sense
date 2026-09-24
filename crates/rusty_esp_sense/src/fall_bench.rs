//! The fall detector's evidence: false alarms on real captures with no falls
//! in them, and the state machine on splices of real wander.
//!
//! No labelled ESP32 fall recording under a usable licence has been found,
//! so detection is not measured on real falls. What is measured:
//!
//! 1. **The burst threshold's floor**, from the TUNING half of the Cuenca
//!    captures (odd positions within each scenario): the highest one-second
//!    wander the on-chip detector reports in any of them -- walking, traffic
//!    or empty, since none of them is a fall.
//! 2. **False alarms** of the chip's `FallDetector` on the TEST half, at the
//!    threshold under test, per hour of real channel state -- and the
//!    threshold at which the first false alarm on the test half appears, so
//!    the margin is a number too.
//! 3. **The state machine on splices** (synthetic, and called so): real
//!    walking wander, a burst, real empty-room wander -- must raise an event;
//!    walking then empty with no burst (someone leaving) must not.

use rusty_esp_signal_core::esp_core::Micros;
use rusty_esp_signal_core::radar::csi::{Config as DetectorConfig, PresenceDetector};
use rusty_esp_signal_core::radar::fall::{FallConfig, FallDetector};

use crate::bench::{Recording, Scenario};
use crate::prof::{self, Counter, Stage};

/// One capture's wander stream: `(at, permille)` per frame.
pub type Stream = Vec<(Micros, u16)>;

/// The on-chip detector's wander, frame by frame.
#[must_use]
pub fn wander_stream(r: &Recording) -> Stream {
    let _g = prof::scope(Stage::Detector);
    prof::add(Counter::DetectorPushes, r.capture.samples.len() as u64);
    prof::add(Counter::FeatureComputations, r.capture.samples.len() as u64);
    let mut det = PresenceDetector::<50>::new(DetectorConfig::normalised_default());
    let mut out = Vec::with_capacity(r.capture.samples.len());
    for s in &r.capture.samples {
        let Ok(f) = s.features() else { continue };
        det.push(&f.normalised(), s.at);
        // The window must fill before the wander means anything.
        if det.warm() {
            out.push((s.at, det.wander()));
        }
    }
    out
}

/// Events a detector with `config` raises over `stream`.
#[must_use]
pub fn events(stream: &Stream, config: FallConfig) -> usize {
    let _g = prof::scope(Stage::Fall);
    prof::add(Counter::FallPushes, stream.len() as u64);
    let mut d = FallDetector::new(config);
    stream
        .iter()
        .filter(|(t, w)| d.push(*w, *t).is_some())
        .count()
}

/// The `len` consecutive frames of `stream` with the most at or above
/// `active`.
fn most_active(stream: &Stream, len: usize, active: u16) -> &[(Micros, u16)] {
    prof::add(Counter::ActiveScans, 1);
    if stream.len() <= len {
        return stream;
    }
    let hot = |w: u16| usize::from(w >= active);
    let mut count: usize = stream[..len].iter().map(|x| hot(x.1)).sum();
    let (mut best, mut at) = (count, 0);
    for i in len..stream.len() {
        count = count + hot(stream[i].1) - hot(stream[i - len].1);
        if count > best {
            best = count;
            at = i + 1 - len;
        }
    }
    &stream[at..at + len]
}

/// What the fall benchmark found.
#[derive(Debug, Clone)]
pub struct FallReport {
    /// Highest wander per scenario on the tuning half.
    pub tuning_max: Vec<(Scenario, u16)>,
    /// The threshold under test.
    pub burst: u16,
    /// False events on the test half at `burst`.
    pub false_events: usize,
    /// Hours of channel state in the test half.
    pub test_hours: f64,
    /// The highest threshold at which the test half raises any event (0 if
    /// none down to the presence threshold).
    pub first_false_at: u16,
    /// Splices with a burst that raised an event, of how many.
    pub splice_falls: (usize, usize),
    /// Splices without a burst (leaving) that raised one, of how many.
    pub splice_leaves: (usize, usize),
}

fn scenario_rank(s: Scenario) -> usize {
    Scenario::ALL.iter().position(|&x| x == s).unwrap_or(0)
}

/// Run it. `burst` of `None` uses [`FallConfig::normalised_default`].
#[must_use]
pub fn run(recordings: &[Recording], burst: Option<u16>) -> FallReport {
    let mut seen = [0usize; 4];
    let mut tuning: Vec<(Scenario, Stream)> = Vec::new();
    let mut test: Vec<(Scenario, Stream)> = Vec::new();
    for r in recordings {
        let i = scenario_rank(r.scenario);
        let stream = wander_stream(r);
        if seen[i] % 2 == 1 {
            tuning.push((r.scenario, stream));
        } else {
            test.push((r.scenario, stream));
        }
        seen[i] += 1;
    }
    let tuning_max = Scenario::ALL
        .iter()
        .map(|&s| {
            let m = tuning
                .iter()
                .filter(|(x, _)| *x == s)
                .flat_map(|(_, st)| st.iter().map(|(_, w)| *w))
                .max()
                .unwrap_or(0);
            (s, m)
        })
        .collect();
    let mut config = FallConfig::normalised_default();
    if let Some(b) = burst {
        config.burst_permille = b;
    }
    let false_events: usize = test.iter().map(|(_, st)| events(st, config)).sum();
    let test_us: u64 = test
        .iter()
        .filter_map(|(_, st)| Some(st.last()?.0.0.saturating_sub(st.first()?.0.0)))
        .sum();
    // The margin: walk the threshold down until the test half alarms.
    let mut first_false_at = 0u16;
    let mut b = config.burst_permille;
    while b > config.active_permille {
        let c = FallConfig {
            burst_permille: b,
            ..config
        };
        if test.iter().any(|(_, st)| events(st, c) > 0) {
            first_false_at = b;
            break;
        }
        b -= 1;
    }
    // Splices: each test walking capture's MOST ACTIVE 10 s of wander, then
    // (or not) a half-second burst above the threshold, then 20 s of a test
    // empty capture's wander, re-timed to follow on. Most active, not first:
    // the labels are per capture and a walker is not always in the path --
    // one capture's first 10 s had no frame above the motion threshold, and
    // a burst out of stillness is correctly not a fall.
    let walks: Vec<&Stream> = test
        .iter()
        .filter(|(s, _)| *s == Scenario::Walking)
        .map(|(_, st)| st)
        .collect();
    let empties: Vec<&Stream> = test
        .iter()
        .filter(|(s, _)| *s == Scenario::Baseline)
        .map(|(_, st)| st)
        .collect();
    let mut splice_falls = (0, 0);
    let mut splice_leaves = (0, 0);
    for (k, walk) in walks.iter().enumerate() {
        let Some(empty) = empties.get(k % empties.len().max(1)) else {
            break;
        };
        for with_burst in [true, false] {
            let mut st: Stream = most_active(walk, 500, config.active_permille).to_vec();
            let mut t = st.last().map_or(0, |(t, _)| t.0);
            if with_burst {
                for _ in 0..25 {
                    t += 20_000;
                    st.push((Micros(t), config.burst_permille.saturating_add(40)));
                }
            }
            let base = empty.first().map_or(0, |(t, _)| t.0);
            for &(te, w) in empty.iter().take(1000) {
                st.push((Micros(t + 20_000 + te.0 - base), w));
            }
            let raised = events(&st, config) > 0;
            let tally = if with_burst {
                &mut splice_falls
            } else {
                &mut splice_leaves
            };
            tally.1 += 1;
            if raised {
                tally.0 += 1;
            }
        }
    }
    FallReport {
        tuning_max,
        burst: config.burst_permille,
        false_events,
        test_hours: test_us as f64 / 3.6e9,
        first_false_at,
        splice_falls,
        splice_leaves,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stream(levels: &[(u64, u16)]) -> Stream {
        let mut out = Vec::new();
        let mut k = 0u64;
        for &(secs, w) in levels {
            for _ in 0..secs * 50 {
                out.push((Micros(k * 20_000), w));
                k += 1;
            }
        }
        out
    }

    #[test]
    fn the_most_active_stretch_is_where_the_motion_is() {
        let s = stream(&[(3, 10), (2, 90), (3, 10)]);
        let seg = most_active(&s, 100, 32);
        assert!(seg.iter().all(|x| x.1 == 90), "{:?}", &seg[..3]);
    }

    #[test]
    fn events_counts_what_the_chip_detector_raises() {
        let fall = stream(&[(5, 60), (1, 400), (15, 18)]);
        assert_eq!(events(&fall, FallConfig::normalised_default()), 1);
        let leave = stream(&[(5, 60), (15, 18)]);
        assert_eq!(events(&leave, FallConfig::normalised_default()), 0);
    }
}
