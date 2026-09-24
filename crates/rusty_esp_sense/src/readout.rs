//! The readout: ridge regression, closed form. Fitting it is the
//! calibration.
//!
//! Features are standardised per column over the calibration data, the
//! targets centred, and `(ZᵀZ + λI) β = ZᵀY` solved by Cholesky with
//! `λ = alpha · n`. The Gram matrix is a candle matmul in f64; the solve is
//! written here, because candle has no linear solve and a `D × D` system
//! for `D` in the low thousands is milliseconds of plain loops.
//! Classification is the same fit on `±1` one-hot targets.

use candle_core::{DType, Device, Tensor};

use crate::prof::{self, Counter, Stage};

/// Per-column mean and standard deviation.
#[derive(Debug, Clone, PartialEq)]
pub struct Standardize {
    /// Column means.
    pub mean: Vec<f32>,
    /// Column standard deviations; a constant column's is 1.
    pub std: Vec<f32>,
}

impl Standardize {
    /// The statistics of `x` (`[n, d]`).
    ///
    /// # Errors
    ///
    /// A tensor error when `x` is not two-dimensional.
    pub fn fit(x: &Tensor) -> crate::Result<Self> {
        let _g = prof::scope(Stage::InputStd);
        let x = x.to_dtype(DType::F64)?;
        let n = x.dim(0)? as f64;
        let mean = (x.sum(0)? / n)?;
        let var = (x.broadcast_sub(&mean)?.sqr()?.sum(0)? / n)?;
        let mean: Vec<f64> = mean.to_vec1()?;
        let var: Vec<f64> = var.to_vec1()?;
        Ok(Standardize {
            mean: mean.iter().map(|&m| m as f32).collect(),
            std: var
                .iter()
                .map(|&v| if v > 1e-12 { v.sqrt() as f32 } else { 1.0 })
                .collect(),
        })
    }

    /// `(x - mean) / std`, for an `[n, d]` tensor.
    ///
    /// One pass into one buffer: the same two `f32` operations per element,
    /// in the same order, as the broadcast subtract and divide it replaces --
    /// which built a mean tensor, a std tensor and an intermediate of the
    /// whole input on every call.
    ///
    /// # Errors
    ///
    /// A tensor error when `x`'s width is not this one's.
    pub fn apply(&self, x: &Tensor) -> crate::Result<Tensor> {
        let _g = prof::scope(Stage::InputStd);
        let (n, d) = x.dims2()?;
        if d != self.mean.len() {
            return Err(crate::Error::Input(format!(
                "standardise: {d} columns, fitted on {}",
                self.mean.len()
            )));
        }
        let mut v: Vec<f32> = x.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        for row in v.chunks_exact_mut(d) {
            for ((e, &m), &s) in row.iter_mut().zip(&self.mean).zip(&self.std) {
                *e = (*e - m) / s;
            }
        }
        Ok(Tensor::from_vec(v, (n, d), &Device::Cpu)?)
    }
}

/// A fitted ridge readout.
#[derive(Debug, Clone, PartialEq)]
pub struct Ridge {
    /// How the features were standardised.
    pub norm: Standardize,
    /// `d × k`, row-major.
    pub beta: Vec<f32>,
    /// The targets' means, `k`.
    pub intercept: Vec<f32>,
    /// Outputs.
    pub outputs: usize,
}

impl Ridge {
    /// Fit `y` (`n × k`, row-major) from `phi` (`[n, d]`).
    ///
    /// # Errors
    ///
    /// [`crate::Error::Input`] when the shapes disagree or the system is not
    /// positive definite (which `alpha > 0` prevents); a tensor error
    /// otherwise.
    pub fn fit(phi: &Tensor, y: &[f32], outputs: usize, alpha: f64) -> crate::Result<Self> {
        let (n, d) = phi.dims2()?;
        if n == 0 || outputs == 0 || y.len() != n * outputs {
            return Err(crate::Error::Input(format!(
                "ridge: {n} rows, {} targets for {outputs} outputs",
                y.len()
            )));
        }
        let norm = Standardize::fit(phi)?;
        let z = norm.apply(phi)?.to_dtype(DType::F64)?;
        let intercept: Vec<f64> = (0..outputs)
            .map(|j| (0..n).map(|i| f64::from(y[i * outputs + j])).sum::<f64>() / n as f64)
            .collect();
        let yc: Vec<f64> = (0..n * outputs)
            .map(|k| f64::from(y[k]) - intercept[k % outputs])
            .collect();
        let yc = Tensor::from_vec(yc, (n, outputs), &Device::Cpu)?;
        let (mut gram, rhs) = {
            let _g = prof::scope(Stage::Gram);
            prof::add(Counter::GramMacs, (n * d * d + n * d * outputs) as u64);
            // The transposed VIEW: the multiply reads it by stride, so the
            // full transposed copy `.contiguous()` made is not needed.
            let zt = z.t()?;
            let gram: Vec<f64> = zt.matmul(&z)?.flatten_all()?.to_vec1()?;
            let rhs: Vec<f64> = zt.matmul(&yc)?.flatten_all()?.to_vec1()?;
            (gram, rhs)
        };
        let lambda = alpha * n as f64;
        for i in 0..d {
            gram[i * d + i] += lambda;
        }
        let beta = cholesky_solve(&mut gram, d, &rhs, outputs)
            .ok_or_else(|| crate::Error::Input("ridge: not positive definite".into()))?;
        Ok(Ridge {
            norm,
            beta: beta.iter().map(|&b| b as f32).collect(),
            intercept: intercept.iter().map(|&m| m as f32).collect(),
            outputs,
        })
    }

    /// The readout's outputs for `phi` (`[n, d]`), `[n, k]`.
    ///
    /// # Errors
    ///
    /// A tensor error when `phi`'s width is not the fitted one.
    pub fn predict(&self, phi: &Tensor) -> crate::Result<Tensor> {
        let z = self.norm.apply(&phi.to_dtype(DType::F32)?)?;
        let _g = prof::scope(Stage::Predict);
        prof::add(Counter::TensorBuilds, 2);
        prof::add(
            Counter::PredictMacs,
            (z.dim(0)? * self.norm.mean.len() * self.outputs) as u64,
        );
        let d = self.norm.mean.len();
        let beta = Tensor::from_slice(&self.beta, (d, self.outputs), &Device::Cpu)?;
        let icpt = Tensor::from_slice(&self.intercept, self.outputs, &Device::Cpu)?;
        Ok(z.matmul(&beta)?.broadcast_add(&icpt)?)
    }
}

/// Solve `A X = B` for symmetric positive definite `A` (`d × d`, row-major,
/// overwritten with its Cholesky factor) and `B` (`d × k`). `None` when `A`
/// is not positive definite.
fn cholesky_solve(a: &mut [f64], d: usize, b: &[f64], k: usize) -> Option<Vec<f64>> {
    let _g = prof::scope(Stage::Cholesky);
    // Inner-loop multiply-adds, exactly: the factor's plus two triangular
    // solves per right-hand side.
    let factor: u64 = (0..d as u64).map(|j| j + (d as u64 - j - 1) * j).sum();
    prof::add(
        Counter::CholeskyMacs,
        factor + (k as u64) * (d as u64) * (d as u64).saturating_sub(1),
    );
    // A = L Lᵀ, L in the lower triangle.
    for j in 0..d {
        let mut s = a[j * d + j];
        for p in 0..j {
            s -= a[j * d + p] * a[j * d + p];
        }
        if s <= 0.0 || !s.is_finite() {
            return None;
        }
        let ljj = s.sqrt();
        a[j * d + j] = ljj;
        for i in j + 1..d {
            let mut s = a[i * d + j];
            for p in 0..j {
                s -= a[i * d + p] * a[j * d + p];
            }
            a[i * d + j] = s / ljj;
        }
    }
    let mut x = b.to_vec();
    for c in 0..k {
        // L y = b
        for i in 0..d {
            let mut s = x[i * k + c];
            for p in 0..i {
                s -= a[i * d + p] * x[p * k + c];
            }
            x[i * k + c] = s / a[i * d + i];
        }
        // Lᵀ x = y
        for i in (0..d).rev() {
            let mut s = x[i * k + c];
            for p in i + 1..d {
                s -= a[p * d + i] * x[p * k + c];
            }
            x[i * k + c] = s / a[i * d + i];
        }
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The cache sweep (codec-analyzer #3): nanoseconds per multiply-add of
    /// the solve as the factor outgrows L1, L2 and L3. Flat means compute;
    /// climbing means memory. Timing, so best-of-N; not a gate.
    #[test]
    #[ignore = "probe: run with --ignored --nocapture, release"]
    fn probe_cholesky_cache_sweep() {
        for d in [64usize, 128, 256, 512, 768, 1024, 1536, 2048] {
            // A = M Mᵀ/d + I, symmetric positive definite, deterministic.
            let m: Vec<f64> = (0..d * d)
                .map(|i| ((i * 2_654_435_761) % 1000) as f64 / 1000.0 - 0.5)
                .collect();
            let mut a0 = vec![0.0; d * d];
            for i in 0..d {
                for j in 0..=i {
                    let s: f64 =
                        (0..d).map(|p| m[i * d + p] * m[j * d + p]).sum::<f64>() / d as f64;
                    a0[i * d + j] = s;
                    a0[j * d + i] = s;
                }
                a0[i * d + i] += 1.0;
            }
            let b: Vec<f64> = (0..2 * d).map(|i| (i % 7) as f64).collect();
            let macs: f64 = (0..d).map(|j| (j + (d - j - 1) * j) as f64).sum::<f64>()
                + 2.0 * (d * (d - 1)) as f64;
            let mut best = f64::MAX;
            let reps = if d <= 512 { 7 } else { 3 };
            for _ in 0..reps {
                let mut a = a0.clone();
                let t = std::time::Instant::now();
                let x = cholesky_solve(&mut a, d, &b, 2).unwrap();
                let ns = t.elapsed().as_nanos() as f64;
                std::hint::black_box(x);
                best = best.min(ns);
            }
            println!(
                "probe d={d:5} working set {:>8} KiB  {:.3} ns/MAC",
                d * d * 8 / 1024,
                best / macs
            );
        }
    }

    #[test]
    fn cholesky_solves_a_known_system() {
        // A = [[4,2],[2,3]], x = [1,2] -> b = [8,8]
        let mut a = vec![4.0, 2.0, 2.0, 3.0];
        let x = cholesky_solve(&mut a, 2, &[8.0, 8.0], 1).unwrap();
        assert!(
            (x[0] - 1.0).abs() < 1e-12 && (x[1] - 2.0).abs() < 1e-12,
            "{x:?}"
        );
        let mut not_pd = vec![1.0, 2.0, 2.0, 1.0];
        assert!(cholesky_solve(&mut not_pd, 2, &[1.0, 1.0], 1).is_none());
    }

    #[test]
    fn ridge_recovers_a_linear_map_and_its_offset() {
        // y0 = 2 a - b + 3, y1 = a + 5 b, exactly, over a spread of points.
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for i in 0..60 {
            let a = (i % 7) as f32 - 3.0;
            let b = (i % 11) as f32 * 0.5;
            xs.extend([a, b]);
            ys.extend([2.0 * a - b + 3.0, a + 5.0 * b]);
        }
        let x = Tensor::from_vec(xs.clone(), (60, 2), &Device::Cpu).unwrap();
        let r = Ridge::fit(&x, &ys, 2, 1e-9).unwrap();
        let p = r.predict(&x).unwrap().to_vec2::<f32>().unwrap();
        for (i, row) in p.iter().enumerate() {
            assert!((row[0] - ys[2 * i]).abs() < 1e-3, "{row:?}");
            assert!((row[1] - ys[2 * i + 1]).abs() < 1e-3, "{row:?}");
        }
    }

    #[test]
    fn a_constant_column_does_not_break_the_fit() {
        let x =
            Tensor::from_vec(vec![1f32, 5.0, 2.0, 5.0, 3.0, 5.0], (3, 2), &Device::Cpu).unwrap();
        let r = Ridge::fit(&x, &[1.0, 2.0, 3.0], 1, 1e-6).unwrap();
        assert_eq!(r.norm.std[1], 1.0);
        let p = r.predict(&x).unwrap().to_vec2::<f32>().unwrap();
        assert!((p[2][0] - 3.0).abs() < 1e-3);
    }

    #[test]
    fn mismatched_targets_are_refused() {
        let x = Tensor::zeros((3, 2), DType::F32, &Device::Cpu).unwrap();
        assert!(Ridge::fit(&x, &[1.0, 2.0], 1, 1.0).is_err());
    }
}
