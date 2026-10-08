//! The stage profiler and the work census (`--features profile`).
//!
//! Two instruments, both off by default and zero-cost when off:
//!
//! - **Stages**: wall time and call counts per pipeline stage, from an RAII
//!   scope at the top of each. Read the ranking and the call counts; a
//!   stage entered millions of times carries its own timer tax.
//! - **Counters**: deterministic work -- frames windowed, per-frame vectors
//!   allocated, bytes copied, detector frames pushed, random values
//!   generated, multiply-adds. Same numbers on any machine under any load,
//!   so a win judged on them is a verdict, not a sample.
//!
//! The binary adds an allocation census (a counting allocator) when built
//! with the feature; the library never declares an allocator.

/// A pipeline stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum Stage {
    /// Reading and parsing a capture.
    Parse,
    /// Cutting windows (per-frame features, normalisation, reduction).
    Window,
    /// The on-chip presence detector, run on the host.
    Detector,
    /// The breathing estimator, run on the host.
    Vitals,
    /// Copying windows into one tensor.
    Stack,
    /// Input standardisation (fit and apply).
    InputStd,
    /// Generating the random encoder.
    EncoderGen,
    /// The encoder's matmul and ReLU.
    Encode,
    /// The readout's standardisation (fit and apply).
    RidgeStd,
    /// `ZᵀZ` and `ZᵀY`.
    Gram,
    /// The Cholesky solve.
    Cholesky,
    /// The readout's prediction.
    Predict,
    /// Argmax over scores.
    Classify,
    /// The fall detector's state machine over wander streams.
    Fall,
    /// Sleep epochs and scoring.
    Night,
    /// Saving and loading models.
    Io,
    /// Gathering windows into a training set (the benchmark's folds).
    Gather,
    /// Everything, from the top of a command.
    Total,
}

/// Stages, in report order.
pub const STAGES: [Stage; 18] = [
    Stage::Parse,
    Stage::Window,
    Stage::Detector,
    Stage::Vitals,
    Stage::Stack,
    Stage::InputStd,
    Stage::EncoderGen,
    Stage::Encode,
    Stage::RidgeStd,
    Stage::Gram,
    Stage::Cholesky,
    Stage::Predict,
    Stage::Classify,
    Stage::Fall,
    Stage::Night,
    Stage::Io,
    Stage::Gather,
    Stage::Total,
];

/// A unit of deterministic work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum Counter {
    /// CSV rows parsed.
    Rows,
    /// Frames that entered a window.
    FramesWindowed,
    /// Per-frame vectors allocated while windowing.
    FrameVecs,
    /// Windows produced.
    Windows,
    /// Bytes of window data cloned to build a training set.
    WindowCloneBytes,
    /// Bytes copied into stacked tensors.
    StackBytes,
    /// Frames pushed through the presence detector.
    DetectorPushes,
    /// Frames pushed through the breathing estimator.
    VitalsPushes,
    /// Random values generated for encoders.
    RandomValues,
    /// Encoder multiply-adds.
    EncodeMacs,
    /// Gram multiply-adds.
    GramMacs,
    /// Cholesky inner-loop multiply-adds (factor and solves).
    CholeskyMacs,
    /// Fall-detector frames pushed.
    FallPushes,
    /// Samples copied (re-timing, splicing).
    SampleCopies,
    /// Tensors built from host slices.
    TensorBuilds,
    /// `CsiFrame::features` computations (per frame, per consumer).
    FeatureComputations,
    /// Bytes validated as UTF-8 on the way in.
    Utf8Bytes,
    /// Fields parsed through the generic `str::parse`.
    FieldParses,
    /// Readout prediction multiply-adds.
    PredictMacs,
    /// Most-active-stretch scans in the fall bench's splices.
    ActiveScans,
}

/// Counters, in report order.
pub const COUNTERS: [Counter; 20] = [
    Counter::Rows,
    Counter::FramesWindowed,
    Counter::FrameVecs,
    Counter::Windows,
    Counter::WindowCloneBytes,
    Counter::StackBytes,
    Counter::DetectorPushes,
    Counter::VitalsPushes,
    Counter::RandomValues,
    Counter::EncodeMacs,
    Counter::GramMacs,
    Counter::CholeskyMacs,
    Counter::FallPushes,
    Counter::SampleCopies,
    Counter::TensorBuilds,
    Counter::FeatureComputations,
    Counter::Utf8Bytes,
    Counter::FieldParses,
    Counter::PredictMacs,
    Counter::ActiveScans,
];

#[cfg(feature = "profile")]
mod imp {
    use core::sync::atomic::{AtomicU64, Ordering::Relaxed};
    use std::time::Instant;

    use super::{COUNTERS, Counter, STAGES, Stage};

    static NS: [AtomicU64; 18] = [const { AtomicU64::new(0) }; 18];
    static CALLS: [AtomicU64; 18] = [const { AtomicU64::new(0) }; 18];
    static COUNT: [AtomicU64; 20] = [const { AtomicU64::new(0) }; 20];

    /// Times a stage while alive.
    pub struct Guard(Stage, Instant);

    impl Drop for Guard {
        fn drop(&mut self) {
            let ns = u64::try_from(self.1.elapsed().as_nanos()).unwrap_or(u64::MAX);
            NS[self.0 as usize].fetch_add(ns, Relaxed);
            CALLS[self.0 as usize].fetch_add(1, Relaxed);
        }
    }

    /// Start timing `s`.
    #[inline]
    #[must_use]
    pub fn scope(s: Stage) -> Guard {
        Guard(s, Instant::now())
    }

    /// Add `n` to a counter.
    #[inline]
    pub fn add(c: Counter, n: u64) {
        COUNT[c as usize].fetch_add(n, Relaxed);
    }

    /// A counter's value.
    #[must_use]
    pub fn get(c: Counter) -> u64 {
        COUNT[c as usize].load(Relaxed)
    }

    /// The report.
    #[must_use]
    pub fn dump() -> String {
        let total = NS[Stage::Total as usize].load(Relaxed).max(1) as f64;
        // Stages add up time from every thread (CPU time); Total is the
        // top of the command (wall time). With parallel steps the stages
        // can sum past Total, and the residue then reads zero.
        let mut s = String::from(
            "stage            ms        %      calls   (stages: summed over threads; Total: wall)\n",
        );
        let mut named = 0u64;
        for st in STAGES {
            let ns = NS[st as usize].load(Relaxed);
            let calls = CALLS[st as usize].load(Relaxed);
            if calls == 0 {
                continue;
            }
            if st != Stage::Total {
                named += ns;
            }
            s.push_str(&format!(
                "{:<14} {:>8.1} {:>7.1} {:>10}\n",
                format!("{st:?}"),
                ns as f64 / 1e6,
                100.0 * ns as f64 / total,
                calls
            ));
        }
        let residue = NS[Stage::Total as usize]
            .load(Relaxed)
            .saturating_sub(named);
        s.push_str(&format!(
            "{:<14} {:>8.1} {:>7.1}\n",
            "(residue)",
            residue as f64 / 1e6,
            100.0 * residue as f64 / total
        ));
        s.push_str("counter                     value\n");
        for c in COUNTERS {
            let v = COUNT[c as usize].load(Relaxed);
            if v > 0 {
                s.push_str(&format!("{:<20} {:>12}\n", format!("{c:?}"), v));
            }
        }
        s
    }
}

#[cfg(not(feature = "profile"))]
mod imp {
    use super::{Counter, Stage};

    /// Nothing, when the feature is off.
    pub struct Guard;

    /// Nothing, when the feature is off.
    #[inline(always)]
    #[must_use]
    pub fn scope(_: Stage) -> Guard {
        Guard
    }

    /// Nothing, when the feature is off.
    #[inline(always)]
    pub fn add(_: Counter, _: u64) {}

    /// Always 0 when the feature is off.
    #[must_use]
    pub fn get(_: Counter) -> u64 {
        0
    }

    /// Empty when the feature is off.
    #[must_use]
    pub fn dump() -> String {
        String::new()
    }
}

pub use imp::{Guard, add, dump, get, scope};
