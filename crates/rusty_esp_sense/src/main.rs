//! `rusty_esp_sense`: calibrate a room, read a recording, or run the
//! Cuenca benchmark.
//!
//! ```text
//! rusty_esp_sense bench-cuenca <dataset-dir> [--frames N] [--features D] [--alpha A] [--folds K] [--fit-jobs J] [--no-centre] [--raw-window]
//! rusty_esp_sense fit --out room.safetensors [--layout L] [--frames N] [--features D] [--alpha A] [--no-centre] [--raw-window] LABEL=a.csv[,b.csv] …
//! rusty_esp_sense run --model room.safetensors [--layout L] recording.csv
//! rusty_esp_sense watch --model room.safetensors [--bridge] [--for SECS] <janus1 ticket>   (feature `live`)
//! ```
//!
//! Recordings are the ledger's fixture format -- what `rusty_esp_iroh-host`'s
//! reference client writes as `csi.csv` from a live W5 stream. The file does
//! not say which training field was captured, so `--layout` does: `lltf`
//! (the default, and what a C10 sends), `ht`, `c6` (the C6's natural HT20
//! order, the Cuenca dataset's), or `dense`.
//!
//! Every command takes `--threads N`: the size of the pool that parses
//! captures, prepares recordings, runs the benchmark's fits side by side and
//! runs the matrix multiplies (the default is one per logical CPU, or
//! `RAYON_NUM_THREADS`). Results do not depend on it -- only time and peak
//! memory do; `bench-cuenca --fit-jobs J` bounds how many fits hold their
//! buffers at once.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use rayon::prelude::*;

use rusty_esp_sense::bench::{self, BenchConfig, Table};
use rusty_esp_sense::capture;
use rusty_esp_sense::model::{FitConfig, Model};
use rusty_esp_sense::window::{self, WindowConfig};
use rusty_esp_signal_core::radar::csi_stream::{
    TAG_C6_HT20_NATURAL, TAG_DENSE_64, TAG_HTLTF_20MHZ, TAG_LLTF_20MHZ,
};

const USAGE: &str = "usage:
  rusty_esp_sense bench-cuenca <dataset-dir> [--frames N] [--features D] [--alpha A] [--folds K] [--fit-jobs J] [--no-centre] [--raw-window]
  rusty_esp_sense fit --out room.safetensors [--layout lltf|ht|c6|dense] [--frames N] [--features D] [--alpha A] [--no-centre] [--raw-window] LABEL=a.csv[,b.csv] ...
  rusty_esp_sense run --model room.safetensors [--layout lltf|ht|c6|dense] recording.csv
  rusty_esp_sense bench-fall <dataset-dir> [--burst PERMILLE]
  rusty_esp_sense night [--layout lltf|ht|c6|dense] recording.csv
  rusty_esp_sense bench-night <dataset-dir>
  rusty_esp_sense watch --model room.safetensors [--bridge] [--for SECS] <janus1 ticket>   (built with --features live)
every command: [--threads N]  (default: one per logical CPU; results do not depend on it)";

struct Args {
    rest: Vec<String>,
}

impl Args {
    fn take(&mut self, flag: &str) -> Option<String> {
        let i = self.rest.iter().position(|a| a == flag)?;
        if i + 1 >= self.rest.len() {
            return None;
        }
        let v = self.rest.remove(i + 1);
        self.rest.remove(i);
        Some(v)
    }

    fn flag(&mut self, flag: &str) -> bool {
        match self.rest.iter().position(|a| a == flag) {
            Some(i) => {
                self.rest.remove(i);
                true
            }
            None => false,
        }
    }

    fn num<T: core::str::FromStr>(&mut self, flag: &str, default: T) -> Result<T, String> {
        match self.take(flag) {
            Some(v) => v.parse().map_err(|_| format!("{flag} {v}: not a number")),
            None => Ok(default),
        }
    }

    fn window(&mut self) -> Result<WindowConfig, String> {
        Ok(WindowConfig {
            frames: self.num("--frames", WindowConfig::DEFAULT.frames)?,
            centre: !self.flag("--no-centre"),
            // the library's default (wander, since the benchmark); --raw-window
            // is the ablation
            wander: WindowConfig::DEFAULT.wander && !self.flag("--raw-window"),
        })
    }

    fn fit(&mut self) -> Result<FitConfig, String> {
        Ok(FitConfig {
            features: self.num("--features", FitConfig::DEFAULT.features)?,
            alpha: self.num("--alpha", FitConfig::DEFAULT.alpha)?,
            ..FitConfig::DEFAULT
        })
    }

    fn layout(&mut self) -> Result<u8, String> {
        match self.take("--layout").as_deref() {
            None | Some("lltf") => Ok(TAG_LLTF_20MHZ),
            Some("ht") => Ok(TAG_HTLTF_20MHZ),
            Some("c6") => Ok(TAG_C6_HT20_NATURAL),
            Some("dense") => Ok(TAG_DENSE_64),
            Some(other) => Err(format!("--layout {other}: lltf, ht, c6 or dense")),
        }
    }
}

/// The allocation census (`--features profile`): a counting wrapper around
/// the system allocator, in the binary only -- a library never declares an
/// allocator. Dev-only, so the one `unsafe` it needs is gated with it.
#[cfg(feature = "profile")]
mod census {
    use core::sync::atomic::{AtomicU64, Ordering::Relaxed};
    use std::alloc::{GlobalAlloc, Layout, System};

    pub static ALLOCS: AtomicU64 = AtomicU64::new(0);
    pub static BYTES: AtomicU64 = AtomicU64::new(0);
    pub static LIVE: AtomicU64 = AtomicU64::new(0);
    pub static PEAK: AtomicU64 = AtomicU64::new(0);

    pub struct Counting;

    // SAFETY: every call forwards to `System` with the caller's own layout
    // and pointer, unchanged; the counters are atomics and never touch the
    // memory handed out.
    #[allow(unsafe_code)]
    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, l: Layout) -> *mut u8 {
            ALLOCS.fetch_add(1, Relaxed);
            BYTES.fetch_add(l.size() as u64, Relaxed);
            let live = LIVE.fetch_add(l.size() as u64, Relaxed) + l.size() as u64;
            PEAK.fetch_max(live, Relaxed);
            // SAFETY: the caller's contract, passed through.
            unsafe { System.alloc(l) }
        }
        unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
            LIVE.fetch_sub(l.size() as u64, Relaxed);
            // SAFETY: the caller's contract, passed through.
            unsafe { System.dealloc(p, l) }
        }
    }

    #[global_allocator]
    static A: Counting = Counting;

    pub fn report() -> String {
        format!(
            "allocations {} bytes {} peak-live {}\n",
            ALLOCS.load(Relaxed),
            BYTES.load(Relaxed),
            PEAK.load(Relaxed)
        )
    }
}

fn main() -> ExitCode {
    let code = {
        let _g = rusty_esp_sense::prof::scope(rusty_esp_sense::prof::Stage::Total);
        real_main()
    };
    #[cfg(feature = "profile")]
    eprint!("{}{}", rusty_esp_sense::prof::dump(), census::report());
    code
}

fn real_main() -> ExitCode {
    let mut all: Vec<String> = std::env::args().skip(1).collect();
    if all.is_empty() {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    }
    let cmd = all.remove(0);
    let mut args = Args { rest: all };
    match args.num("--threads", 0usize) {
        // The one pool every parallel step and every multiply runs on.
        Ok(0) => {}
        Ok(n) => {
            if let Err(e) = rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build_global()
            {
                eprintln!("rusty_esp_sense: --threads {n}: {e}");
                return ExitCode::FAILURE;
            }
        }
        Err(e) => {
            eprintln!("rusty_esp_sense: {e}");
            return ExitCode::from(2);
        }
    }
    let result = match cmd.as_str() {
        "bench-cuenca" => bench_cuenca(&mut args),
        "fit" => fit(&mut args),
        "run" => run(&mut args),
        "bench-fall" => bench_fall(&mut args),
        "night" => night(&mut args),
        "bench-night" => bench_night(&mut args),
        #[cfg(feature = "live")]
        "watch" => watch(&mut args),
        _ => Err(USAGE.to_owned()),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("rusty_esp_sense: {e}");
            ExitCode::FAILURE
        }
    }
}

fn pct(x: f64) -> String {
    format!("{:5.1} %", 100.0 * x)
}

fn print_table(title: &str, t: &Table) {
    println!("\n{title}");
    println!(
        "  {:<22} {:>8} {:>10} {:>10} {:>12}",
        "scenario", "windows", "occupied", "accuracy", "captures ok"
    );
    for (s, row) in t {
        let captures_right = if s.occupied() {
            row.captures_called_occupied
        } else {
            row.captures - row.captures_called_occupied
        };
        println!(
            "  {:<22} {:>8} {:>10} {:>10} {:>12}",
            s.name(),
            row.windows,
            pct(row.called_occupied as f64 / row.windows.max(1) as f64),
            pct(row.accuracy(*s)),
            format!("{captures_right}/{}", row.captures),
        );
    }
    println!("  balanced accuracy {}", pct(bench::balanced(t)));
}

fn bench_cuenca(args: &mut Args) -> Result<(), String> {
    let cfg = BenchConfig {
        window: args.window()?,
        fit: args.fit()?,
        folds: args.num("--folds", 5)?,
        fit_jobs: args.num("--fit-jobs", 0)?,
    };
    let dir = args.rest.first().cloned().ok_or_else(|| USAGE.to_owned())?;
    let started = std::time::Instant::now();
    let recs = bench::load(Path::new(&dir)).map_err(|e| e.to_string())?;
    let r = bench::run(&recs, &cfg).map_err(|e| e.to_string())?;
    println!(
        "Cuenca benchmark: {} captures, {} windows each (median) of {} frames ({}{}), {} rows refused",
        recs.len(),
        r.windows_per_capture,
        cfg.window.frames,
        if cfg.window.centre {
            "centred"
        } else {
            "not centred"
        },
        if cfg.window.wander {
            ", per-subcarrier wander"
        } else {
            ""
        },
        r.rejected_rows
    );
    println!(
        "model: {} random features, seed {:#x}, alpha {}; {}-fold, whole captures held out; model file {} B",
        cfg.fit.features, cfg.fit.seed, cfg.fit.alpha, cfg.folds, r.model_bytes
    );
    for (s, days) in &r.days {
        println!("  {} recorded on {}", s.name(), days.join(", "));
    }
    print_table(
        "held out (every capture scored by a fit that never saw it)",
        &r.held_out,
    );
    print_table(
        "the on-chip detector (W1b, gain-normalised; no training)",
        &r.detector,
    );
    print_table(
        "trained on E1 + E2 only, scored on E3 + E4 (the day confound, the traffic false alarm)",
        &r.confound,
    );
    println!("\n{:.1} s", started.elapsed().as_secs_f64());
    Ok(())
}

fn fit(args: &mut Args) -> Result<(), String> {
    let out = args.take("--out").ok_or("fit needs --out")?;
    let layout = args.layout()?;
    let win = args.window()?;
    let cfg = args.fit()?;
    // The label specs, in order, name the files; a malformed spec ends the
    // list there, as it ended the sequential loop. Each file's read and
    // windows depend on that file alone, so they run in parallel; the
    // checks, the messages and the training set then follow in file order,
    // and the first failure in that order is the one returned.
    let mut labels: Vec<String> = Vec::new();
    let mut jobs: Vec<(usize, &str, &str)> = Vec::new();
    let mut bad_spec = None;
    for spec in &args.rest {
        let Some((label, files)) = spec.split_once('=') else {
            bad_spec = Some(format!("{spec}: LABEL=file.csv[,file.csv]"));
            break;
        };
        let idx = match labels.iter().position(|l| l == label) {
            Some(i) => i,
            None => {
                labels.push(label.to_owned());
                labels.len() - 1
            }
        };
        for f in files.split(',') {
            jobs.push((idx, label, f));
        }
    }
    // Each file's windows in one buffer (a heap vector per window before);
    // the training set then borrows their rows.
    let read: Vec<Result<(usize, window::FlatWindows), String>> = jobs
        .par_iter()
        .map(|&(_, _, f)| {
            let c = capture::read(&PathBuf::from(f), layout, bench::FRAME_US)
                .map_err(|e| format!("{f}: {e}"))?;
            Ok((c.rejected, window::windows_flat(&c.samples, win)))
        })
        .collect();
    let mut files = Vec::with_capacity(read.len());
    let mut targets = Vec::new();
    let mut subcarriers = 0usize;
    for (&(idx, label, f), r) in jobs.iter().zip(read) {
        let (rejected, w) = r?;
        if subcarriers == 0 {
            subcarriers = w.subcarriers;
        }
        if w.subcarriers != subcarriers {
            return Err(format!(
                "{f}: {} subcarriers, the rest {subcarriers}",
                w.subcarriers
            ));
        }
        eprintln!(
            "{label}: {f}: {} windows ({rejected} rows refused, {} frames skipped)",
            w.count, w.skipped
        );
        targets.extend(std::iter::repeat_n(idx, w.count));
        files.push(w);
    }
    if let Some(e) = bad_spec {
        return Err(e);
    }
    let mut data: Vec<&[f32]> = Vec::with_capacity(targets.len());
    for w in &files {
        data.extend(w.rows());
    }
    let model =
        Model::fit(&data, &targets, labels, subcarriers, win, cfg).map_err(|e| e.to_string())?;
    model.save(Path::new(&out)).map_err(|e| e.to_string())?;
    let bytes = std::fs::metadata(&out).map(|m| m.len()).unwrap_or(0);
    eprintln!("calibrated on {} windows: {out} ({bytes} B)", data.len());
    Ok(())
}

fn run(args: &mut Args) -> Result<(), String> {
    let path = args.take("--model").ok_or("run needs --model")?;
    let layout = args.layout()?;
    let model = Model::load(Path::new(&path)).map_err(|e| e.to_string())?;
    let f = args.rest.first().ok_or_else(|| USAGE.to_owned())?;
    let c = capture::read(Path::new(f), layout, bench::FRAME_US).map_err(|e| e.to_string())?;
    let w = window::windows(&c.samples, model.window);
    if w.subcarriers != model.subcarriers {
        return Err(format!(
            "{f}: {} subcarriers; the model was calibrated on {} (check --layout)",
            w.subcarriers, model.subcarriers
        ));
    }
    let scores = model.scores(&w.data).map_err(|e| e.to_string())?;
    let secs = model.window.frames as f64 * bench::FRAME_US as f64 / 1e6;
    for (i, s) in scores.iter().enumerate() {
        let best = s
            .iter()
            .enumerate()
            .fold(
                (0, f32::NEG_INFINITY),
                |b, (j, &v)| if v > b.1 { (j, v) } else { b },
            )
            .0;
        println!(
            "{}",
            serde_json::json!({
                "window": i,
                "at_s": i as f64 * secs,
                "label": model.labels[best],
                "scores": model.labels.iter().zip(s).map(|(l, v)| (l.clone(), serde_json::Value::from(f64::from(*v)))).collect::<serde_json::Map<String, serde_json::Value>>(),
            })
        );
    }
    Ok(())
}

/// Subscribe to a W5 stream and read the room live: one JSON line per
/// window per device, the same windows and readout `run` uses on a file.
/// A device's own stream is `"csi "`; a bridge's is `"nbrc"`, whose packets
/// name each neighbour, so one bridge can carry several rooms' devices and
/// each is windowed on its own.
#[cfg(feature = "live")]
fn watch(args: &mut Args) -> Result<(), String> {
    use std::collections::HashMap;
    use std::time::Duration;

    use rusty_esp_iroh_host::client::endpoint_addr;
    use rusty_esp_iroh_host::core::media::Subscribe;
    use rusty_esp_iroh_host::core::telemetry::{CODEC_CSI, CODEC_NEIGHBOUR_CSI};
    use rusty_esp_iroh_host::core::ticket::Ticket;
    use rusty_esp_iroh_host::{Client, csi};
    use rusty_esp_signal_core::radar::csi_stream::Sample;

    let path = args.take("--model").ok_or("watch needs --model")?;
    let bridge = args.flag("--bridge");
    let secs: u64 = args.num("--for", 3600)?;
    let model = Model::load(Path::new(&path)).map_err(|e| e.to_string())?;
    let text = args.rest.first().ok_or_else(|| USAGE.to_owned())?.clone();
    let ticket = Ticket::parse_text(&text).map_err(|e| format!("ticket: {e:?}"))?;
    let addr = endpoint_addr(&ticket).map_err(|e| e.to_string())?;
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| e.to_string())?;
    rt.block_on(async {
        let client = Client::bind(None, None, ticket.relay().is_some())
            .await
            .map_err(|e| e.to_string())?;
        // A device's own "csi " packets carry no DID; its manifest names it.
        let own_did = if bridge {
            String::new()
        } else {
            client.manifest(&addr).await.map(|m| m.did).unwrap_or_default()
        };
        let sub = Subscribe {
            codec: if bridge { CODEC_NEIGHBOUR_CSI } else { CODEC_CSI },
            max_fps: 0,
            max_kbps: 0,
        };
        let mut pending: HashMap<String, Vec<Sample>> = HashMap::new();
        let mut refused = 0u64;
        let counter = client
            .subscribe(&addr, &sub, u64::MAX, Duration::from_secs(secs), |h, payload| {
                let Some(r) = csi::from_packet(h.codec, payload, &own_did) else {
                    refused += 1;
                    return;
                };
                let buf = pending.entry(r.did.clone()).or_default();
                buf.push(r.sample);
                if buf.len() < model.window.frames {
                    return;
                }
                let w = window::windows(buf, model.window);
                let at_us = buf.last().map_or(0, |s| s.at.0);
                buf.clear();
                if w.subcarriers != model.subcarriers || w.data.is_empty() {
                    eprintln!(
                        "{}: {} subcarriers, the model was calibrated on {}",
                        r.did, w.subcarriers, model.subcarriers
                    );
                    return;
                }
                if let Ok(scores) = model.scores(&w.data) {
                    for s in scores {
                        let best = s
                            .iter()
                            .enumerate()
                            .fold((0, f32::NEG_INFINITY), |b, (j, &v)| if v > b.1 { (j, v) } else { b })
                            .0;
                        println!(
                            "{}",
                            serde_json::json!({
                                "did": r.did,
                                "reach": r.reach,
                                "at_us": at_us,
                                "label": model.labels[best],
                                "scores": model.labels.iter().zip(&s).map(|(l, v)| (l.clone(), serde_json::Value::from(f64::from(*v)))).collect::<serde_json::Map<String, serde_json::Value>>(),
                            })
                        );
                    }
                }
            })
            .await
            .map_err(|e| e.to_string())?;
        eprintln!(
            "watched {secs} s: received={} lost={} reordered={} not-a-sample={refused}",
            counter.received, counter.lost, counter.reordered
        );
        Ok::<(), String>(())
    })
}

fn bench_fall(args: &mut Args) -> Result<(), String> {
    let burst: Option<u16> = match args.take("--burst") {
        Some(v) => Some(
            v.parse()
                .map_err(|_| format!("--burst {v}: not a number"))?,
        ),
        None => None,
    };
    let dir = args.rest.first().cloned().ok_or_else(|| USAGE.to_owned())?;
    let recs = bench::load(Path::new(&dir)).map_err(|e| e.to_string())?;
    let r = rusty_esp_sense::fall_bench::run(&recs, burst);
    println!(
        "fall benchmark: {} captures, split in halves within each scenario",
        recs.len()
    );
    println!("tuning half, the highest one-second wander the on-chip detector reports:");
    for (s, m) in &r.tuning_max {
        println!("  {:<22} {m} permille", s.name());
    }
    println!(
        "test half: {:.2} h of channel state, burst threshold {} permille: {} false events",
        r.test_hours, r.burst, r.false_events
    );
    if r.first_false_at > 0 {
        println!(
            "  the first false event on the test half appears at a threshold of {} permille",
            r.first_false_at
        );
    } else {
        println!(
            "  no false event on the test half at any threshold down to the presence threshold"
        );
    }
    println!(
        "splices (SYNTHETIC: real walking wander, a half-second burst, real empty-room wander): {}/{} raised",
        r.splice_falls.0, r.splice_falls.1
    );
    println!(
        "splices with no burst (walking, then an empty room: someone leaving): {}/{} raised",
        r.splice_leaves.0, r.splice_leaves.1
    );
    Ok(())
}

fn night(args: &mut Args) -> Result<(), String> {
    use rusty_esp_sense::sleep::{self, NightConfig};
    let layout = args.layout()?;
    let f = args.rest.first().ok_or_else(|| USAGE.to_owned())?;
    let c = capture::read(Path::new(f), layout, bench::FRAME_US).map_err(|e| e.to_string())?;
    let cfg = NightConfig::DEFAULT;
    let epochs = sleep::epochs(&c.samples, &cfg);
    let states = sleep::score(&epochs, &cfg);
    for (e, s) in epochs.iter().zip(&states) {
        println!(
            "{}",
            serde_json::json!({
                "start_s": e.start.0 as f64 / 1e6,
                "state": s.word(),
                "motion": e.motion,
                "breathing_bpm": e.breathing_bpm_x10.map(|b| f64::from(b) / 10.0),
                "breathing_confidence": e.breathing_confidence,
            })
        );
    }
    let sum = sleep::summarise(&epochs, &states, &cfg);
    println!(
        "{}",
        serde_json::json!({
            "summary": {
                "epochs": sum.epochs,
                "in_room_min": sum.in_room_min,
                "asleep_min": sum.asleep_min,
                "onset_min": sum.onset_min,
                "awake_after_onset_min": sum.awake_after_onset_min,
                "efficiency": sum.efficiency,
                "awakenings": sum.awakenings,
                "asleep_breathing_bpm": sum.asleep_breathing_bpm,
            }
        })
    );
    Ok(())
}

/// Each Cuenca scenario played back as one continuous stretch -- its
/// captures end to end -- and scored as a night. None of them is a night:
/// the point is what an empty room and a walking person are scored AS. An
/// empty room scored asleep is the failure that matters.
fn bench_night(args: &mut Args) -> Result<(), String> {
    use rusty_esp_sense::bench::Scenario;
    use rusty_esp_sense::sleep::{self, NightConfig, State};
    use rusty_esp_signal_core::esp_core::Micros;
    let dir = args.rest.first().cloned().ok_or_else(|| USAGE.to_owned())?;
    let recs = bench::load(Path::new(&dir)).map_err(|e| e.to_string())?;
    let cfg = NightConfig::DEFAULT;
    println!(
        "night benchmark: each scenario's captures played end to end, scored in {} s epochs",
        cfg.epoch.0 / 1_000_000
    );
    println!("  scenario                epochs   empty   awake  asleep  breathing accepted");
    // Each scenario is its own night, played end to end through stateful
    // estimators: a stream cannot be split, but the four streams are
    // independent. They run side by side and print in scenario order.
    let nights: Vec<_> = Scenario::ALL
        .par_iter()
        .map(|&s| {
            // Played end to end by re-timing each capture to follow the last --
            // on the fly: the samples were copied into a new vector (42 MB)
            // only to be read once.
            let caps: Vec<_> = recs.iter().filter(|r| r.scenario == s).collect();
            let mut offsets = Vec::with_capacity(caps.len());
            let mut offset = 0u64;
            for r in &caps {
                offsets.push(offset);
                let base = r.capture.samples.first().map_or(0, |x| x.at.0);
                let end = r
                    .capture
                    .samples
                    .last()
                    .map_or(offset, |x| offset + (x.at.0 - base));
                offset = end + bench::FRAME_US;
            }
            // Each sample read at its new time, by reference: copying the
            // whole record to change its timestamp moved 42 MB.
            let retimed = caps.iter().zip(offsets).flat_map(|(r, off)| {
                let base = r.capture.samples.first().map_or(0, |x| x.at.0);
                r.capture
                    .samples
                    .iter()
                    .map(move |x| (Micros(off + (x.at.0 - base)), x))
            });
            let epochs = sleep::epochs_at(retimed, &cfg);
            let states = sleep::score(&epochs, &cfg);
            (s, epochs, states)
        })
        .collect();
    for (s, epochs, states) in nights {
        let count = |k: State| states.iter().filter(|&&x| x == k).count();
        let breathed = epochs
            .iter()
            .filter(|e| e.breathing_bpm_x10.is_some())
            .count();
        println!(
            "  {:<22} {:>7} {:>7} {:>7} {:>7}  {breathed}/{}",
            s.name(),
            states.len(),
            count(State::Empty),
            count(State::Awake),
            count(State::Asleep),
            epochs.len()
        );
    }
    Ok(())
}
