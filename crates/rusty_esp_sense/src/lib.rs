//! Janus W6: the home computer's model over Wi-Fi channel state.
//!
//! The device (W0-W5) runs the fixed-point detector and ships every frame's
//! raw I/Q ([`rusty_esp_signal_core::radar::csi_stream::Sample`]). This crate
//! is what reads that stream downstream -- a home computer or any LAN box --
//! and turns windows of it into a room state: occupied or not today, a pose
//! when a labelled rig exists.
//!
//! # Why random features and a per-room readout
//!
//! The RuView plan's first W6 step was to run RuView's released weights on
//! our stream. Their own model card and study rule that out three ways:
//! the weights take `[3 antennas, 114 subcarriers, 10 frames]` at 100 Hz
//! from MM-Fi's radio (an ESP32 gives one antenna and 52-56 subcarriers at
//! 50 Hz); they are CC BY-NC 4.0, which a commercial deployment cannot use;
//! and their study measures cross-environment zero-shot at ~10 % and finds
//! that a *random frozen* encoder with a trained readout comes within 2-4
//! points of a fully trained one. Channel state is distribution-locked: what
//! transfers is the method, not the weights, and the signal lives in a
//! readout fitted where the device is.
//!
//! So the model here is exactly that, and nothing more:
//!
//! 1. [`window`]: non-overlapping windows of gain-normalised amplitude per
//!    subcarrier (W1b: a receiver's gain step is not the room), optionally
//!    centred in time so the window carries change, not the room's static
//!    shape.
//! 2. [`encoder`]: a fixed random projection and a ReLU, regenerated from a
//!    seed -- nothing about it is learned, so nothing about it is shipped
//!    beyond the seed.
//! 3. [`readout`]: ridge regression, closed form, fitted per room from a
//!    labelled clip. That fit IS the calibration.
//!
//! [`model::Model`] is the three together and saves as one safetensors file
//! of a few kilobytes. [`bench`] measures it on the Cuenca dataset with
//! whole captures held out, beside the on-chip detector.

pub mod bench;
pub mod capture;
pub mod encoder;
pub mod fall_bench;
pub mod metrics;
pub mod model;
pub mod prof;
pub mod readout;
pub mod sleep;
pub mod window;

/// What can go wrong.
#[derive(Debug)]
pub enum Error {
    /// A tensor operation failed.
    Tensor(candle_core::Error),
    /// Input that cannot be used, with why.
    Input(String),
    /// A model file that is not one of ours.
    Model(String),
    /// Reading or writing a file.
    Io(std::io::Error),
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::Tensor(e) => write!(f, "tensor: {e}"),
            Error::Input(s) => write!(f, "input: {s}"),
            Error::Model(s) => write!(f, "model: {s}"),
            Error::Io(e) => write!(f, "io: {e}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<candle_core::Error> for Error {
    fn from(e: candle_core::Error) -> Self {
        Error::Tensor(e)
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

/// This crate's results.
pub type Result<T> = core::result::Result<T, Error>;
