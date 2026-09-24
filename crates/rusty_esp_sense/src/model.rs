//! The model: windows → standardised → fixed random features → ridge
//! readout. One safetensors file holds everything a machine needs to read a
//! room the way this one was calibrated.
//!
//! The encoder is stored as its seed and widths, never its weights: the
//! generator is this crate's own, pinned by a test, so the same seed is the
//! same encoder everywhere. What is stored is what was fitted to the room --
//! the input statistics and the readout -- and that is the calibration.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Device, Tensor};

use crate::encoder::RandomFeatures;
use crate::prof::{self, Stage};
use crate::readout::{Ridge, Standardize};
use crate::window::WindowConfig;

/// The file format's version, in the `meta` tensor.
pub const FORMAT: i64 = 1;

/// How to fit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FitConfig {
    /// Random features; 0 fits the readout on the standardised windows
    /// directly (the ablation RuView's §7 invites).
    pub features: usize,
    /// The encoder's seed.
    pub seed: u64,
    /// Ridge strength, as a fraction of the calibration set's size.
    pub alpha: f64,
}

impl FitConfig {
    /// 1024 random features, seed `"JANUS"`, `alpha` 0.01 -- chosen before
    /// any held-out data was scored, and not changed since. (The WINDOW
    /// default was changed after the benchmark; `WindowConfig::DEFAULT`
    /// says so.)
    pub const DEFAULT: FitConfig = FitConfig {
        features: 1024,
        seed: 0x4A_414E_5553,
        alpha: 0.01,
    };
}

impl Default for FitConfig {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// A calibrated model.
#[derive(Debug, Clone)]
pub struct Model {
    /// How windows are cut.
    pub window: WindowConfig,
    /// Subcarriers per frame it was calibrated on.
    pub subcarriers: usize,
    /// Per-input statistics of the calibration windows.
    pub input: Standardize,
    /// The fixed encoder, or `None` for a readout on the inputs.
    pub encoder: Option<RandomFeatures>,
    /// The fitted readout.
    pub ridge: Ridge,
    /// Class names, in output order.
    pub labels: Vec<String>,
}

impl Model {
    /// Calibrate a classifier: `data` windows (all `subcarriers ×
    /// window.frames` wide), `targets` indexes into `labels`.
    ///
    /// # Errors
    ///
    /// [`crate::Error::Input`] when there is nothing to fit, a target is out
    /// of range, or a window has the wrong width; a tensor error otherwise.
    pub fn fit<R: AsRef<[f32]>>(
        data: &[R],
        targets: &[usize],
        labels: Vec<String>,
        subcarriers: usize,
        window: WindowConfig,
        cfg: FitConfig,
    ) -> crate::Result<Self> {
        Self::fit_with(data, targets, labels, subcarriers, window, cfg, None)
    }

    /// [`Model::fit`], borrowing an encoder already generated for the same
    /// seed and widths. The encoder is a pure function of those three
    /// numbers, so a caller fitting many models (the benchmark's folds)
    /// generates it once; one that does not match is ignored and a fresh
    /// one generated, so a wrong hint costs time, never correctness.
    ///
    /// # Errors
    ///
    /// As [`Model::fit`].
    pub fn fit_with<R: AsRef<[f32]>>(
        data: &[R],
        targets: &[usize],
        labels: Vec<String>,
        subcarriers: usize,
        window: WindowConfig,
        cfg: FitConfig,
        encoder: Option<&RandomFeatures>,
    ) -> crate::Result<Self> {
        let k = labels.len();
        if data.is_empty() || data.len() != targets.len() || k < 2 {
            return Err(crate::Error::Input(format!(
                "{} windows, {} targets, {k} labels",
                data.len(),
                targets.len()
            )));
        }
        if let Some(t) = targets.iter().find(|&&t| t >= k) {
            return Err(crate::Error::Input(format!("target {t} with {k} labels")));
        }
        let width = window.width(subcarriers);
        let encoder = match encoder {
            _ if cfg.features == 0 => None,
            Some(e) if e.seed == cfg.seed && e.input == width && e.output == cfg.features => {
                Some(e.clone())
            }
            _ => Some(RandomFeatures::new(cfg.seed, width, cfg.features)?),
        };
        // The stacked and standardised inputs live only until encoded, not
        // through the readout's fit.
        let (input, phi) = {
            if data.iter().any(|w| w.as_ref().len() != width) {
                return Err(crate::Error::Input(format!(
                    "a window is not {width} values wide"
                )));
            }
            let input = Standardize::of_rows(data, width);
            let z = input.stack_apply(data)?;
            let phi = match &encoder {
                Some(e) => e.encode(&z)?,
                None => z,
            };
            (input, phi)
        };
        // One-hot, +1 / -1, k per window: sized once (a flattening
        // iterator's size hint starts at zero, so collecting it doubled).
        let mut y: Vec<f32> = Vec::with_capacity(targets.len() * k);
        for &t in targets {
            y.extend((0..k).map(|j| if j == t { 1.0 } else { -1.0 }));
        }
        let ridge = Ridge::fit(&phi, &y, k, cfg.alpha)?;
        Ok(Model {
            window,
            subcarriers,
            input,
            encoder,
            ridge,
            labels,
        })
    }

    /// The readout's scores per window, one per label.
    ///
    /// # Errors
    ///
    /// [`crate::Error::Input`] when a window has the wrong width.
    pub fn scores<R: AsRef<[f32]>>(&self, data: &[R]) -> crate::Result<Vec<Vec<f32>>> {
        let k = self.ridge.outputs.max(1);
        Ok(self
            .scores_flat(data)?
            .chunks(k)
            .map(<[f32]>::to_vec)
            .collect())
    }

    /// [`Model::scores`], row-major in one buffer (`windows × labels`): what
    /// a caller that only compares scores wants, without a vector per
    /// window.
    fn scores_flat<R: AsRef<[f32]>>(&self, data: &[R]) -> crate::Result<Vec<f32>> {
        if data.is_empty() {
            return Ok(Vec::new());
        }
        let width = self.window.width(self.subcarriers);
        if data.iter().any(|w| w.as_ref().len() != width) {
            return Err(crate::Error::Input(format!(
                "a window is not {width} values wide"
            )));
        }
        let z = self.input.stack_apply(data)?;
        let phi = match &self.encoder {
            Some(e) => e.encode(&z)?,
            None => z,
        };
        // The features were made here and nothing else holds them.
        let y = self.ridge.predict_owned(phi)?;
        let _g = prof::scope(Stage::Predict);
        Ok(y.flatten_all()?.to_vec1()?)
    }

    /// The label index each window reads as.
    ///
    /// # Errors
    ///
    /// As [`Model::scores`].
    pub fn classify<R: AsRef<[f32]>>(&self, data: &[R]) -> crate::Result<Vec<usize>> {
        let scores = self.scores_flat(data)?;
        let _g = prof::scope(Stage::Classify);
        Ok(scores
            .chunks(self.ridge.outputs.max(1))
            .map(|s| {
                s.iter()
                    .enumerate()
                    .fold(
                        (0, f32::NEG_INFINITY),
                        |b, (i, &v)| if v > b.1 { (i, v) } else { b },
                    )
                    .0
            })
            .collect())
    }

    /// Save as safetensors.
    ///
    /// # Errors
    ///
    /// A tensor or I/O error.
    pub fn save(&self, path: &Path) -> crate::Result<()> {
        let _g = prof::scope(Stage::Io);
        let dev = Device::Cpu;
        let (features, seed) = self.encoder.as_ref().map_or((0, 0), |e| (e.output, e.seed));
        let meta: Vec<i64> = vec![
            FORMAT,
            self.window.frames as i64,
            i64::from(self.window.centre) | (i64::from(self.window.wander) << 1),
            self.subcarriers as i64,
            features as i64,
            seed as i64,
            self.ridge.outputs as i64,
        ];
        let d = self.ridge.norm.mean.len();
        let mut t: HashMap<String, Tensor> = HashMap::new();
        t.insert("meta".into(), Tensor::from_vec(meta, 7, &dev)?);
        t.insert(
            "labels".into(),
            Tensor::from_vec(
                self.labels.join("\n").into_bytes(),
                self.labels.join("\n").len(),
                &dev,
            )?,
        );
        let v = |x: &[f32]| Tensor::from_slice(x, x.len(), &dev);
        t.insert("input.mean".into(), v(&self.input.mean)?);
        t.insert("input.std".into(), v(&self.input.std)?);
        t.insert("ridge.mean".into(), v(&self.ridge.norm.mean)?);
        t.insert("ridge.std".into(), v(&self.ridge.norm.std)?);
        t.insert(
            "ridge.beta".into(),
            Tensor::from_slice(&self.ridge.beta, (d, self.ridge.outputs), &dev)?,
        );
        t.insert("ridge.intercept".into(), v(&self.ridge.intercept)?);
        candle_core::safetensors::save(&t, path)?;
        Ok(())
    }

    /// Load what [`Model::save`] wrote.
    ///
    /// # Errors
    ///
    /// [`crate::Error::Model`] for a file that is not one of ours or a
    /// format version this build does not know; a tensor error otherwise.
    pub fn load(path: &Path) -> crate::Result<Self> {
        let t = {
            let _g = prof::scope(Stage::Io);
            candle_core::safetensors::load(path, &Device::Cpu)?
        };
        let get = |k: &str| {
            t.get(k)
                .ok_or_else(|| crate::Error::Model(format!("no `{k}` in {}", path.display())))
        };
        let f32s = |k: &str| -> crate::Result<Vec<f32>> {
            Ok(get(k)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?)
        };
        let meta: Vec<i64> = get("meta")?.to_vec1()?;
        if meta.len() != 7 || meta[0] != FORMAT {
            return Err(crate::Error::Model(format!("format {:?}", meta.first())));
        }
        let usize_of = |v: i64| usize::try_from(v).map_err(|_| crate::Error::Model(format!("{v}")));
        let window = WindowConfig {
            frames: usize_of(meta[1])?,
            centre: meta[2] & 1 != 0,
            wander: meta[2] & 2 != 0,
        };
        let subcarriers = usize_of(meta[3])?;
        let features = usize_of(meta[4])?;
        let seed = meta[5] as u64;
        let outputs = usize_of(meta[6])?;
        let labels_bytes: Vec<u8> = get("labels")?.to_vec1()?;
        let labels: Vec<String> = String::from_utf8(labels_bytes)
            .map_err(|_| crate::Error::Model("labels are not UTF-8".into()))?
            .split('\n')
            .map(str::to_owned)
            .collect();
        let width = window.width(subcarriers);
        let encoder = if features > 0 {
            Some(RandomFeatures::new(seed, width, features)?)
        } else {
            None
        };
        let ridge = Ridge {
            norm: Standardize {
                mean: f32s("ridge.mean")?,
                std: f32s("ridge.std")?,
            },
            beta: f32s("ridge.beta")?,
            intercept: f32s("ridge.intercept")?,
            outputs,
        };
        if labels.len() != outputs || ridge.beta.len() != ridge.norm.mean.len() * outputs {
            return Err(crate::Error::Model("shapes disagree".into()));
        }
        Ok(Model {
            window,
            subcarriers,
            input: Standardize {
                mean: f32s("input.mean")?,
                std: f32s("input.std")?,
            },
            encoder,
            ridge,
            labels,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two classes: a still window, and one where the first half of the
    /// subcarriers swing.
    fn data(n: usize) -> (Vec<Vec<f32>>, Vec<usize>) {
        let (s, t) = (8, 10);
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for i in 0..n {
            let moving = i % 2 == 1;
            let mut w = vec![0f32; s * t];
            for k in 0..s {
                for j in 0..t {
                    let noise = (((i * 31 + k * 7 + j * 3) % 13) as f32 - 6.0) * 0.002;
                    let swing = if moving && k < s / 2 {
                        0.1 * ((j as f32) * 0.9 + i as f32).sin()
                    } else {
                        0.0
                    };
                    w[k * t + j] = noise + swing;
                }
            }
            xs.push(w);
            ys.push(usize::from(moving));
        }
        (xs, ys)
    }

    fn labels() -> Vec<String> {
        vec!["empty".into(), "occupied".into()]
    }

    #[test]
    fn a_calibrated_model_separates_what_it_was_shown() {
        let (x, y) = data(80);
        let cfg = FitConfig {
            features: 128,
            ..FitConfig::DEFAULT
        };
        let win = WindowConfig {
            frames: 10,
            centre: true,
            wander: false,
        };
        let m = Model::fit(&x[..60], &y[..60], labels(), 8, win, cfg).unwrap();
        let p = m.classify(&x[60..]).unwrap();
        let right = p.iter().zip(&y[60..]).filter(|(a, b)| a == b).count();
        assert!(right >= 18, "{right}/20");
    }

    #[test]
    fn a_saved_model_loads_and_reads_the_same() {
        let (x, y) = data(40);
        let win = WindowConfig {
            frames: 10,
            centre: true,
            wander: false,
        };
        for features in [0, 64] {
            let cfg = FitConfig {
                features,
                ..FitConfig::DEFAULT
            };
            let m = Model::fit(&x, &y, labels(), 8, win, cfg).unwrap();
            let dir = std::env::temp_dir().join(format!("rusty_esp_sense-{features}.safetensors"));
            m.save(&dir).unwrap();
            let back = Model::load(&dir).unwrap();
            assert_eq!(back.labels, labels());
            assert_eq!(back.window, win);
            assert_eq!(m.scores(&x).unwrap(), back.scores(&x).unwrap());
            let _ = std::fs::remove_file(&dir);
        }
    }

    #[test]
    fn what_cannot_be_fitted_is_refused() {
        let (x, y) = data(4);
        let win = WindowConfig {
            frames: 10,
            centre: true,
            wander: false,
        };
        assert!(Model::fit(&x, &y[..3], labels(), 8, win, FitConfig::DEFAULT).is_err());
        assert!(Model::fit(&x, &[0, 1, 2, 0], labels(), 8, win, FitConfig::DEFAULT).is_err());
        assert!(
            Model::fit(&x, &y, labels(), 9, win, FitConfig::DEFAULT).is_err(),
            "wrong width"
        );
    }
}
