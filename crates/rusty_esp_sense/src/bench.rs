//! The Cuenca benchmark: 100 real ESP32-C6 captures, whole captures held
//! out, beside the on-chip detector.
//!
//! The dataset (Universidad de Cuenca, Zenodo 10.5281/zenodo.21148028, CC BY
//! 4.0) is two ESP32-C6s, 50 Hz injected frames, 60 s per capture, in four
//! scenarios:
//!
//! | scenario | the room | captures |
//! |---|---|---|
//! | E1 | empty, still, no traffic | 15 |
//! | E2 | one person walking across the line of sight | 20 |
//! | E3 | empty, the channel saturated with UDP at 2 / 5 / 10 Mbps | 45 |
//! | E4 | a person walking AND bursty traffic | 20 |
//!
//! E3 is why this dataset is worth more than its size: network traffic
//! moves channel state too, and the paper that published it asks whether
//! the two can be told apart. An "occupied" call on E3 is a false alarm a
//! home would see every time someone streams a film.
//!
//! It also carries a trap, printed in the report rather than hidden: every
//! E1 capture was recorded on one day and every E2/E3/E4 capture on
//! another, so a model can score well on E1-vs-E2 by learning the DAY. The
//! confound experiment trains on E1 and E2 alone and asks what it calls E3
//! -- empty, but recorded on the walking day.
//!
//! Protocol: windows never overlap; every fold holds out whole captures (a
//! capture's windows are never split between training and test); folds are
//! assigned round-robin within each scenario over sorted file names, so the
//! split is the same on every machine. Nothing here is tuned on a test
//! fold: [`crate::model::FitConfig::DEFAULT`] was fixed before any
//! held-out score existed.

use std::path::Path;

use rusty_esp_signal_core::esp_core::Micros;
use rusty_esp_signal_core::radar::csi::{Config as DetectorConfig, PresenceDetector, Verdict};
use rusty_esp_signal_core::radar::csi_stream::TAG_C6_HT20_NATURAL;

use crate::capture::{self, Capture};
use crate::encoder::RandomFeatures;
use crate::model::{FitConfig, Model};
use crate::prof::{self, Counter, Stage};
use crate::readout::{Ridge, Standardize};
use crate::window::{WindowBuilder, WindowConfig, Windows};

/// The dataset's frame interval: 50 Hz.
pub const FRAME_US: u64 = 20_000;

/// One of the four scenarios.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Scenario {
    /// E1: empty, still, no traffic.
    Baseline,
    /// E2: a person walking.
    Walking,
    /// E3: empty, saturated with UDP traffic.
    Traffic,
    /// E4: a person walking and bursty traffic.
    Coexistence,
}

impl Scenario {
    /// All four, in order.
    pub const ALL: [Scenario; 4] = [
        Scenario::Baseline,
        Scenario::Walking,
        Scenario::Traffic,
        Scenario::Coexistence,
    ];

    /// Whether someone is in the room.
    #[must_use]
    pub const fn occupied(self) -> bool {
        matches!(self, Scenario::Walking | Scenario::Coexistence)
    }

    /// The report's name for it.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Scenario::Baseline => "E1 empty",
            Scenario::Walking => "E2 walking",
            Scenario::Traffic => "E3 empty + traffic",
            Scenario::Coexistence => "E4 walking + traffic",
        }
    }

    /// The scenario a capture's file name prefix names; `None` for the
    /// traffic-rate logs (`tasas_*`), which are not channel state.
    #[must_use]
    pub fn of_prefix(prefix: &str) -> Option<Scenario> {
        if prefix.starts_with("linea_base") {
            Some(Scenario::Baseline)
        } else if prefix.starts_with("movimiento_humano") {
            Some(Scenario::Walking)
        } else if prefix.starts_with("trafico_udp") {
            Some(Scenario::Traffic)
        } else if prefix.starts_with("mov_trafico") {
            Some(Scenario::Coexistence)
        } else {
            None
        }
    }
}

/// One capture of the dataset.
#[derive(Debug, Clone)]
pub struct Recording {
    /// Its scenario.
    pub scenario: Scenario,
    /// The day it was recorded, `YYYYMMDD`, from its file name.
    pub day: String,
    /// The rows.
    pub capture: Capture,
}

/// Load every capture under `root` (the dataset's top folder, holding
/// `Escenario 1` … `Escenario 4`), sorted by scenario and file name.
///
/// # Errors
///
/// [`crate::Error::Io`] when a folder or file cannot be read;
/// [`crate::Error::Input`] when no capture was found.
pub fn load(root: &Path) -> crate::Result<Vec<Recording>> {
    let mut out = Vec::new();
    for dir in std::fs::read_dir(root)? {
        let dir = dir?.path();
        if !dir.is_dir() {
            continue;
        }
        for f in std::fs::read_dir(&dir)? {
            let path = f?.path();
            if path.extension().is_none_or(|e| e != "csv") {
                continue;
            }
            let stem = path
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
            let Some((prefix, rest)) = stem.split_once("_iter_") else {
                continue;
            };
            let Some(scenario) = Scenario::of_prefix(prefix) else {
                continue;
            };
            let day = rest.split('_').nth(1).unwrap_or("").to_owned();
            let capture = capture::read(&path, TAG_C6_HT20_NATURAL, FRAME_US)?;
            out.push(Recording {
                scenario,
                day,
                capture,
            });
        }
    }
    if out.is_empty() {
        return Err(crate::Error::Input(format!(
            "no Cuenca captures under {}",
            root.display()
        )));
    }
    out.sort_by(|a, b| (a.scenario, &a.capture.name).cmp(&(b.scenario, &b.capture.name)));
    Ok(out)
}

/// How to run it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BenchConfig {
    /// Windows.
    pub window: WindowConfig,
    /// The model.
    pub fit: FitConfig,
    /// Folds of whole captures.
    pub folds: usize,
}

impl Default for BenchConfig {
    fn default() -> Self {
        BenchConfig {
            window: WindowConfig::DEFAULT,
            fit: FitConfig::DEFAULT,
            folds: 5,
        }
    }
}

/// Calls counted over one scenario.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Tally {
    /// Windows scored.
    pub windows: usize,
    /// Of those, called occupied.
    pub called_occupied: usize,
    /// Captures scored.
    pub captures: usize,
    /// Captures whose majority of windows was called occupied.
    pub captures_called_occupied: usize,
}

impl Tally {
    fn add(&mut self, calls: &[bool]) {
        let occ = calls.iter().filter(|&&c| c).count();
        self.windows += calls.len();
        self.called_occupied += occ;
        self.captures += 1;
        if 2 * occ > calls.len() {
            self.captures_called_occupied += 1;
        }
    }

    /// The fraction of windows called RIGHT for `scenario`.
    #[must_use]
    pub fn accuracy(&self, scenario: Scenario) -> f64 {
        if self.windows == 0 {
            return 0.0;
        }
        let occ = self.called_occupied as f64 / self.windows as f64;
        if scenario.occupied() { occ } else { 1.0 - occ }
    }
}

/// One row per scenario.
pub type Table = Vec<(Scenario, Tally)>;

/// The mean of the per-scenario window accuracies: every scenario weighs
/// the same, however many captures it has.
#[must_use]
pub fn balanced(table: &Table) -> f64 {
    let rows: Vec<f64> = table
        .iter()
        .filter(|(_, t)| t.windows > 0)
        .map(|(s, t)| t.accuracy(*s))
        .collect();
    if rows.is_empty() {
        0.0
    } else {
        rows.iter().sum::<f64>() / rows.len() as f64
    }
}

/// What the benchmark found.
#[derive(Debug, Clone)]
pub struct Report {
    /// The model, every capture scored by a fit that never saw it.
    pub held_out: Table,
    /// The on-chip detector (W1b, gain-normalised, its own thresholds): no
    /// training at all, every capture.
    pub detector: Table,
    /// The model trained on E1 and E2 only (every capture of each), scored
    /// on E3 and E4: the day confound and the traffic false alarm.
    pub confound: Table,
    /// The days each scenario was recorded on.
    pub days: Vec<(Scenario, Vec<String>)>,
    /// Bytes of a model file calibrated on everything.
    pub model_bytes: u64,
    /// Rows the parser refused, over every capture.
    pub rejected_rows: usize,
    /// Windows per capture (the median).
    pub windows_per_capture: usize,
}

struct Prepared {
    scenario: Scenario,
    fold: usize,
    windows: Vec<Vec<f32>>,
    detector: Vec<bool>,
}

/// A capture's windows and, over the same frames, the on-chip detector's
/// call at the end of each window -- in ONE pass. Each frame's features are
/// computed and normalised once and handed to both; they were computed
/// twice before (docs/PERF.md).
fn prepare(c: &Capture, cfg: WindowConfig) -> (Windows, Vec<bool>) {
    let _g = prof::scope(Stage::Window);
    prof::add(Counter::DetectorPushes, c.samples.len() as u64);
    prof::add(Counter::FeatureComputations, c.samples.len() as u64);
    let frames = cfg.frames.max(1);
    let mut det = PresenceDetector::<50>::new(DetectorConfig::normalised_default());
    let mut b = WindowBuilder::new(cfg);
    let mut data = Vec::new();
    let mut calls = Vec::new();
    let mut seen = 0usize;
    for s in &c.samples {
        let norm = s.features().ok().map(|f| f.normalised());
        if let Some(w) = b.push(norm.as_ref()) {
            data.push(w);
        }
        if let Some(f) = &norm {
            let verdict = det.push(f, Micros(s.at.0));
            seen += 1;
            if seen % frames == 0 {
                calls.push(matches!(verdict, Verdict::Present { .. }));
            }
        }
    }
    calls.truncate(data.len());
    let w = Windows {
        subcarriers: b.subcarriers(),
        data,
        skipped: b.skipped(),
    };
    (w, calls)
}

fn table(rows: &[(Scenario, Vec<bool>)]) -> Table {
    Scenario::ALL
        .iter()
        .map(|&s| {
            let mut t = Tally::default();
            for (_, calls) in rows.iter().filter(|(r, _)| *r == s) {
                t.add(calls);
            }
            (s, t)
        })
        .filter(|(_, t)| t.captures > 0)
        .collect()
}

/// Every capture's calls, from ONE scoring pass over all their windows:
/// scored one capture at a time, each call rebuilt the standardisation and
/// readout tensors for the same model. Each window's scores are its own
/// row's, so the split back per capture is exact.
fn called_occupied(model: &Model, captures: &[&Prepared]) -> crate::Result<Vec<Vec<bool>>> {
    let occupied = model
        .labels
        .iter()
        .position(|l| l == "occupied")
        .ok_or_else(|| crate::Error::Model("no `occupied` label".into()))?;
    let rows: Vec<&[f32]> = captures
        .iter()
        .flat_map(|p| p.windows.iter().map(Vec::as_slice))
        .collect();
    let calls = model.classify(&rows)?;
    let mut out = Vec::with_capacity(captures.len());
    let mut at = 0;
    for p in captures {
        let n = p.windows.len();
        out.push(calls[at..at + n].iter().map(|&c| c == occupied).collect());
        at += n;
    }
    Ok(out)
}

fn fit_on(
    prepared: &[&Prepared],
    subcarriers: usize,
    cfg: &BenchConfig,
    encoder: Option<&RandomFeatures>,
) -> crate::Result<Model> {
    let (data, targets) = {
        let _g = prof::scope(Stage::Gather);
        let mut data = Vec::new();
        let mut targets = Vec::new();
        for p in prepared {
            for w in &p.windows {
                prof::add(Counter::WindowCloneBytes, (w.len() * 4) as u64);
                data.push(w.clone());
                targets.push(usize::from(p.scenario.occupied()));
            }
        }
        (data, targets)
    };
    Model::fit_with(
        &data,
        &targets,
        vec!["empty".into(), "occupied".into()],
        subcarriers,
        cfg.window,
        cfg.fit,
        encoder,
    )
}

/// A zero-valued model with exactly the shapes a fit with `cfg` produces.
fn shaped_like(subcarriers: usize, cfg: &BenchConfig, encoder: Option<&RandomFeatures>) -> Model {
    let width = cfg.window.width(subcarriers);
    let d = if cfg.fit.features > 0 {
        cfg.fit.features
    } else {
        width
    };
    let outputs = 2;
    let zeros = |n: usize| vec![0f32; n];
    Model {
        window: cfg.window,
        subcarriers,
        input: Standardize {
            mean: zeros(width),
            std: zeros(width),
        },
        encoder: encoder.cloned(),
        ridge: Ridge {
            norm: Standardize {
                mean: zeros(d),
                std: zeros(d),
            },
            beta: zeros(d * outputs),
            intercept: zeros(outputs),
            outputs,
        },
        labels: vec!["empty".into(), "occupied".into()],
    }
}

/// Run the benchmark over `recordings`.
///
/// # Errors
///
/// [`crate::Error::Input`] when captures disagree on their subcarrier count
/// or a fold has nothing to train on; a model error otherwise.
pub fn run(recordings: &[Recording], cfg: &BenchConfig) -> crate::Result<Report> {
    let folds = cfg.folds.max(2);
    let mut subcarriers = 0usize;
    let mut prepared = Vec::with_capacity(recordings.len());
    let mut per_scenario = [0usize; 4];
    let mut counts = Vec::new();
    for r in recordings {
        let (w, detector) = prepare(&r.capture, cfg.window);
        if subcarriers == 0 {
            subcarriers = w.subcarriers;
        }
        if w.subcarriers != subcarriers {
            return Err(crate::Error::Input(format!(
                "{} has {} subcarriers, the rest {subcarriers}",
                r.capture.name, w.subcarriers
            )));
        }
        let idx = Scenario::ALL
            .iter()
            .position(|&s| s == r.scenario)
            .unwrap_or(0);
        let fold = per_scenario[idx] % folds;
        per_scenario[idx] += 1;
        counts.push(w.data.len());
        prepared.push(Prepared {
            scenario: r.scenario,
            fold,
            windows: w.data,
            detector,
        });
    }

    // One encoder for every fit below: it depends on the seed and the widths
    // only, and was generated seven times before (docs/PERF.md).
    let encoder = if cfg.fit.features > 0 {
        Some(RandomFeatures::new(
            cfg.fit.seed,
            cfg.window.width(subcarriers),
            cfg.fit.features,
        )?)
    } else {
        None
    };

    // Held out, fold by fold.
    let mut held = Vec::new();
    for f in 0..folds {
        let train: Vec<&Prepared> = prepared.iter().filter(|p| p.fold != f).collect();
        let test: Vec<&Prepared> = prepared.iter().filter(|p| p.fold == f).collect();
        if test.is_empty() {
            continue;
        }
        let model = fit_on(&train, subcarriers, cfg, encoder.as_ref())?;
        for (p, calls) in test.iter().zip(called_occupied(&model, &test)?) {
            held.push((p.scenario, calls));
        }
    }

    // The confound: E1 + E2 only.
    let e12: Vec<&Prepared> = prepared
        .iter()
        .filter(|p| matches!(p.scenario, Scenario::Baseline | Scenario::Walking))
        .collect();
    let mut confound = Vec::new();
    if !e12.is_empty() {
        let model = fit_on(&e12, subcarriers, cfg, encoder.as_ref())?;
        let later: Vec<&Prepared> = prepared
            .iter()
            .filter(|p| matches!(p.scenario, Scenario::Traffic | Scenario::Coexistence))
            .collect();
        for (p, calls) in later.iter().zip(called_occupied(&model, &later)?) {
            confound.push((p.scenario, calls));
        }
    }

    // The size of a model file calibrated on everything. A safetensors file's
    // size is a function of its tensors' SHAPES (the header holds shapes and
    // offsets), never their values, so a zero-valued model of the same shape
    // measures it without an eighth fit's Gram matrix and Cholesky solve.
    let model = shaped_like(subcarriers, cfg, encoder.as_ref());
    let path = std::env::temp_dir().join(format!(
        "rusty_esp_sense-bench-{}.safetensors",
        std::process::id()
    ));
    model.save(&path)?;
    let model_bytes = std::fs::metadata(&path)?.len();
    let _ = std::fs::remove_file(&path);

    let detector: Vec<(Scenario, Vec<bool>)> = prepared
        .iter()
        .map(|p| (p.scenario, p.detector.clone()))
        .collect();
    let days = Scenario::ALL
        .iter()
        .map(|&s| {
            let mut d: Vec<String> = recordings
                .iter()
                .filter(|r| r.scenario == s)
                .map(|r| r.day.clone())
                .collect();
            d.sort();
            d.dedup();
            (s, d)
        })
        .collect();
    counts.sort_unstable();
    Ok(Report {
        held_out: table(&held),
        detector: table(&detector),
        confound: table(&confound),
        days,
        model_bytes,
        rejected_rows: recordings.iter().map(|r| r.capture.rejected).sum(),
        windows_per_capture: counts.get(counts.len() / 2).copied().unwrap_or(0),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The oracle for R2: a model fitted on data and the zero-valued one of
    /// the same shape save to files of identical size, raw window or wander,
    /// with or without the encoder.
    #[test]
    fn a_fitted_model_and_its_shape_save_to_the_same_size() {
        for (wander, features) in [(true, 64), (true, 0), (false, 32)] {
            let window = WindowConfig {
                frames: 10,
                centre: true,
                wander,
            };
            let cfg = BenchConfig {
                window,
                fit: FitConfig {
                    features,
                    ..FitConfig::DEFAULT
                },
                folds: 5,
            };
            let s = 6;
            let w = window.width(s);
            let data: Vec<Vec<f32>> = (0..20)
                .map(|i| {
                    (0..w)
                        .map(|j| ((i * 7 + j * 3) % 11) as f32 * 0.01)
                        .collect()
                })
                .collect();
            let targets: Vec<usize> = (0..20).map(|i| i % 2).collect();
            let fitted = Model::fit(
                &data,
                &targets,
                vec!["empty".into(), "occupied".into()],
                s,
                window,
                cfg.fit,
            )
            .unwrap();
            let enc = fitted.encoder.clone();
            let shaped = shaped_like(s, &cfg, enc.as_ref());
            let dir = std::env::temp_dir();
            let (a, b) = (
                dir.join(format!("r2-fit-{wander}-{features}.st")),
                dir.join(format!("r2-shape-{wander}-{features}.st")),
            );
            fitted.save(&a).unwrap();
            shaped.save(&b).unwrap();
            let size = |p: &std::path::Path| std::fs::metadata(p).unwrap().len();
            assert_eq!(size(&a), size(&b), "wander {wander}, features {features}");
            let _ = std::fs::remove_file(&a);
            let _ = std::fs::remove_file(&b);
        }
    }

    #[test]
    fn file_names_name_their_scenario_and_the_rate_logs_are_left_out() {
        assert_eq!(Scenario::of_prefix("linea_base"), Some(Scenario::Baseline));
        assert_eq!(
            Scenario::of_prefix("movimiento_humano"),
            Some(Scenario::Walking)
        );
        assert_eq!(
            Scenario::of_prefix("trafico_udp_10mbps"),
            Some(Scenario::Traffic)
        );
        assert_eq!(
            Scenario::of_prefix("mov_trafico"),
            Some(Scenario::Coexistence)
        );
        assert_eq!(Scenario::of_prefix("tasas"), None);
        assert!(Scenario::Walking.occupied() && Scenario::Coexistence.occupied());
        assert!(!Scenario::Baseline.occupied() && !Scenario::Traffic.occupied());
    }

    #[test]
    fn a_tally_scores_empty_and_occupied_the_right_way_round() {
        let mut t = Tally::default();
        t.add(&[true, true, false, false]);
        t.add(&[true, true, true, false]);
        assert_eq!((t.windows, t.called_occupied), (8, 5));
        assert_eq!(
            (t.captures, t.captures_called_occupied),
            (2, 1),
            "a tie is not a majority"
        );
        assert!((t.accuracy(Scenario::Walking) - 0.625).abs() < 1e-12);
        assert!((t.accuracy(Scenario::Baseline) - 0.375).abs() < 1e-12);
    }
}
