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
        // Column sums in f64, row by row, streamed from the f32 values: no
        // f64 copy of the input, no deviations tensor, no squares tensor.
        let (n, d) = x.dims2()?;
        let v: Vec<f32> = x.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let nf = n as f64;
        let mut mean = vec![0f64; d];
        for row in v.chunks_exact(d) {
            widen_add(&mut mean, row);
        }
        for m in &mut mean {
            *m /= nf;
        }
        let mut var = vec![0f64; d];
        for row in v.chunks_exact(d) {
            widen_add_squared_deviations(&mut var, row, &mean);
        }
        for s in &mut var {
            *s /= nf;
        }
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

/// `acc[c] += row[c] as f64`, every column's accumulator side by side --
/// each still summed over rows in order. Its own frame so the slices arrive
/// as non-aliasing parameters and the loop vectorises; inlined into
/// [`Standardize::fit`] it stayed scalar (the census, as for `wander`).
#[inline(never)]
fn widen_add(acc: &mut [f64], row: &[f32]) {
    for (a, &e) in acc.iter_mut().zip(row) {
        *a += f64::from(e);
    }
}

/// `acc[c] += (row[c] as f64 - mean[c])²`, as [`widen_add`].
#[inline(never)]
fn widen_add_squared_deviations(acc: &mut [f64], row: &[f32], mean: &[f64]) {
    for ((a, &e), &m) in acc.iter_mut().zip(row).zip(mean) {
        let dev = f64::from(e) - m;
        *a += dev * dev;
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
            prof::add(Counter::GramMacs, (n * d * outputs) as u64);
            // The transposed VIEW: the multiply reads it by stride, so the
            // full transposed copy `.contiguous()` made is not needed.
            let zt = z.t()?;
            let gram = lower_gram(&zt, &z, d)?;
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

/// Rows of `ZᵀZ` computed per panel: GRAM_PANEL rows of `Zᵀ` against only
/// the columns up to the panel's end. The Cholesky reads the lower triangle
/// alone, so the rest of the square is never computed; it stays zero.
const GRAM_PANEL: usize = 128;

fn lower_gram(zt: &Tensor, z: &Tensor, d: usize) -> crate::Result<Vec<f64>> {
    let mut gram = vec![0f64; d * d];
    let mut r0 = 0;
    while r0 < d {
        let r1 = (r0 + GRAM_PANEL).min(d);
        prof::add(Counter::GramMacs, (z.dim(0)? * (r1 - r0) * r1) as u64);
        let block: Vec<f64> = zt
            .narrow(0, r0, r1 - r0)?
            .matmul(&z.narrow(1, 0, r1)?)?
            .flatten_all()?
            .to_vec1()?;
        for (row, src) in gram[r0 * d..r1 * d]
            .chunks_exact_mut(d)
            .zip(block.chunks_exact(r1))
        {
            row[..r1].copy_from_slice(src);
        }
        r0 = r1;
    }
    Ok(gram)
}

/// Solve `A X = B` for symmetric positive definite `A` (`d × d`, row-major,
/// overwritten with its Cholesky factor) and `B` (`d × k`). `None` when `A`
/// is not positive definite.
///
/// The kernels below are written for the vector unit and the cache, and
/// every one of them is bit-identical to the plain loops kept as
/// `tests::reference_solve` -- each sum runs in its original order; what
/// is vectorised is independent accumulators side by side, never one sum
/// reassociated.
fn cholesky_solve(a: &mut [f64], d: usize, b: &[f64], k: usize) -> Option<Vec<f64>> {
    let _g = prof::scope(Stage::Cholesky);
    // Inner-loop multiply-adds, exactly: the factor's plus two triangular
    // solves per right-hand side.
    let factor: u64 = (0..d as u64).map(|j| j + (d as u64 - j - 1) * j).sum();
    prof::add(
        Counter::CholeskyMacs,
        factor + (k as u64) * (d as u64) * (d as u64).saturating_sub(1),
    );
    if !cholesky_factor(a, d) {
        return None;
    }
    let mut x = b.to_vec();
    if k == 2 {
        solve_forward_pair(a, d, &mut x);
        solve_back_pair(a, d, &mut x);
    } else {
        for c in 0..k {
            solve_forward(a, d, &mut x, k, c);
            solve_back(a, d, &mut x, k, c);
        }
    }
    Some(x)
}

/// Columns per panel of the blocked factor, and the side of its tile.
const PANEL: usize = 16;

/// `A = L Lᵀ`, `L` in the lower triangle of `a`; `false` when `A` is not
/// positive definite.
///
/// Blocked by panels of [`PANEL`] columns, for the cache: before a panel is
/// finished, the products over every earlier column are subtracted from it
/// one [`PANEL`]-wide block of columns at a time, ascending, the running
/// value stored back between blocks. Each entry therefore sees exactly the
/// subtractions of the unblocked loop in exactly its order -- the blocks
/// only change which entries are worked on while a tile sits in L1.
#[inline(never)]
fn cholesky_factor(a: &mut [f64], d: usize) -> bool {
    // tile[p][jj] = L[j0 + jj][p0 + p]: the panel's rows over one block of
    // earlier columns, packed so sixteen columns load side by side.
    let mut tile = [[0f64; PANEL]; PANEL];
    let mut j0 = 0;
    while j0 < d {
        let j1 = (j0 + PANEL).min(d);
        let w = j1 - j0;
        let mut p0 = 0;
        while p0 < j0 {
            for (p, t) in tile.iter_mut().enumerate() {
                for (jj, v) in t.iter_mut().enumerate() {
                    *v = if jj < w {
                        a[(j0 + jj) * d + p0 + p]
                    } else {
                        0.0
                    };
                }
            }
            for i in j0..d {
                let jn = (i + 1).min(j1) - j0;
                update_row(&mut a[i * d..i * d + j1], p0, j0, jn, &tile);
            }
            p0 += PANEL;
        }
        if !factor_panel(a, d, j0, j1) {
            return false;
        }
        j0 = j1;
    }
    true
}

/// `row[j0 + jj] -= Σ_p row[p0 + p] · tile[p][jj]` for `jj < jn`, the sum
/// over `p` in order: sixteen independent accumulators side by side, each
/// subtracted exactly as the unblocked loop subtracts. Lanes at or past
/// `jn` compute and are dropped.
#[inline(never)]
fn update_row(row: &mut [f64], p0: usize, j0: usize, jn: usize, tile: &[[f64; PANEL]; PANEL]) {
    let mut acc = [0f64; PANEL];
    acc[..jn].copy_from_slice(&row[j0..j0 + jn]);
    let x = &row[p0..p0 + PANEL];
    for (&xp, t) in x.iter().zip(tile) {
        for (s, &y) in acc.iter_mut().zip(t) {
            *s -= xp * y;
        }
    }
    row[j0..j0 + jn].copy_from_slice(&acc[..jn]);
}

/// Finish columns `from..to`: subtract the products over columns
/// `from..j` (everything before `from` was subtracted already, in order),
/// then take the root and divide. With `from = 0, to = d` this is the whole
/// unblocked factor.
#[inline(never)]
fn factor_panel(a: &mut [f64], d: usize, from: usize, to: usize) -> bool {
    for j in from..to {
        // The row's prefix as a slice: the same subtractions in the same
        // order, without a bounds check on each of two indexings per trip.
        let mut s = a[j * d + j];
        for &v in &a[j * d + from..j * d + j] {
            s -= v * v;
        }
        if s <= 0.0 || !s.is_finite() {
            return false;
        }
        let ljj = s.sqrt();
        a[j * d + j] = ljj;
        // Four rows at once: four independent accumulators, each summed in
        // its own row's order exactly as below, sharing each load of row j.
        // Vertical, not reassociated -- bit for bit the one-row loop.
        let mut i = j + 1;
        while i + 4 <= d {
            let s = {
                let rj = &a[j * d + from..j * d + j];
                let r0 = &a[i * d + from..i * d + j];
                let r1 = &a[(i + 1) * d + from..(i + 1) * d + j];
                let r2 = &a[(i + 2) * d + from..(i + 2) * d + j];
                let r3 = &a[(i + 3) * d + from..(i + 3) * d + j];
                let mut s = [
                    a[i * d + j],
                    a[(i + 1) * d + j],
                    a[(i + 2) * d + j],
                    a[(i + 3) * d + j],
                ];
                for ((((&y, &x0), &x1), &x2), &x3) in rj.iter().zip(r0).zip(r1).zip(r2).zip(r3) {
                    s[0] -= x0 * y;
                    s[1] -= x1 * y;
                    s[2] -= x2 * y;
                    s[3] -= x3 * y;
                }
                s
            };
            for (r, v) in s.iter().enumerate() {
                a[(i + r) * d + j] = v / ljj;
            }
            i += 4;
        }
        for i in i..d {
            // Both rows' prefixes as slices, zipped: the same order, no
            // bounds check per trip; the write waits until they are done.
            let s = {
                let (row_i, row_j) = (&a[i * d + from..i * d + j], &a[j * d + from..j * d + j]);
                let mut s = a[i * d + j];
                for (&x, &y) in row_i.iter().zip(row_j) {
                    s -= x * y;
                }
                s
            };
            a[i * d + j] = s / ljj;
        }
    }
    true
}

/// `L y = b` for column `c` of `x` (`d × k`), in place.
#[inline(never)]
fn solve_forward(a: &[f64], d: usize, x: &mut [f64], k: usize, c: usize) {
    for i in 0..d {
        let mut s = x[i * k + c];
        for p in 0..i {
            s -= a[i * d + p] * x[p * k + c];
        }
        x[i * k + c] = s / a[i * d + i];
    }
}

/// `L y = b` for both columns of a two-column `x` (`d × 2`) at once: two
/// independent lanes, each subtracted in its own column's order, sharing
/// each load of `L` -- and each row's pair of `x` is contiguous, so the
/// lanes load, multiply and subtract as one packed pair. Bit for bit two
/// calls of [`solve_forward`]; the two columns' back-substitutions run
/// after, as before, since a column's forward and back solves never read
/// the other column.
#[inline(never)]
fn solve_forward_pair(a: &[f64], d: usize, x: &mut [f64]) {
    for i in 0..d {
        let s = {
            let mut s = [x[2 * i], x[2 * i + 1]];
            for (&l, pair) in a[i * d..i * d + i].iter().zip(x.chunks_exact(2)) {
                s[0] -= l * pair[0];
                s[1] -= l * pair[1];
            }
            s
        };
        let dii = a[i * d + i];
        x[2 * i] = s[0] / dii;
        x[2 * i + 1] = s[1] / dii;
    }
}

/// `Lᵀ x = y` for both columns of a two-column `x` at once, as
/// [`solve_forward_pair`] does forward: bit for bit two calls of
/// [`solve_back`].
#[inline(never)]
fn solve_back_pair(a: &[f64], d: usize, x: &mut [f64]) {
    for i in (0..d).rev() {
        let s = {
            let mut s = [x[2 * i], x[2 * i + 1]];
            // Below row i of column i; for the last row there is nothing
            // below, and the start index is past the matrix.
            let column = a.get((i + 1) * d + i..).unwrap_or(&[]).iter().step_by(d);
            for (&l, pair) in column.zip(x[2 * (i + 1)..].chunks_exact(2)) {
                s[0] -= l * pair[0];
                s[1] -= l * pair[1];
            }
            s
        };
        let dii = a[i * d + i];
        x[2 * i] = s[0] / dii;
        x[2 * i + 1] = s[1] / dii;
    }
}

/// `Lᵀ x = y` for column `c` of `x` (`d × k`), in place.
#[inline(never)]
fn solve_back(a: &[f64], d: usize, x: &mut [f64], k: usize, c: usize) {
    for i in (0..d).rev() {
        let mut s = x[i * k + c];
        for p in i + 1..d {
            s -= a[p * d + i] * x[p * k + c];
        }
        x[i * k + c] = s / a[i * d + i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The plain loops the kernels are held to, bit for bit: the solve as it
    /// was written before any optimisation, kept as the oracle.
    /// Solve `A X = B` for symmetric positive definite `A` (`d × d`, row-major,
    /// overwritten with its Cholesky factor) and `B` (`d × k`). `None` when `A`
    /// is not positive definite.
    fn reference_solve(a: &mut [f64], d: usize, b: &[f64], k: usize) -> Option<Vec<f64>> {
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

    /// Every kernel against the oracle, bit for bit, over sizes that hit
    /// every remainder of every unrolled or blocked loop, and 1-3 columns.
    #[test]
    fn the_kernels_match_the_reference_bit_for_bit() {
        for d in (1usize..=40).chain([63, 64, 65, 127, 128, 129, 200]) {
            let mut a0 = vec![0.0f64; d * d];
            for i in 0..d {
                for j in 0..=i {
                    let v = (((i * 7919 + j * 104_729) % 997) as f64 / 997.0 - 0.5) * 0.9;
                    a0[i * d + j] = v;
                    a0[j * d + i] = v;
                }
                a0[i * d + i] += d as f64 * 0.6 + 1.0;
            }
            for k in 1..=3 {
                let b: Vec<f64> = (0..d * k)
                    .map(|i| ((i * 31) % 17) as f64 - 8.0 + i as f64 / 7.0)
                    .collect();
                let want = reference_solve(&mut a0.clone(), d, &b, k).expect("SPD");
                let got = cholesky_solve(&mut a0.clone(), d, &b, k).expect("SPD");
                let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(&got), bits(&want), "d {d} k {k}");
                let mut fa = a0.clone();
                let mut ra = a0.clone();
                assert!(cholesky_factor(&mut fa, d));
                let _ = reference_solve(&mut ra, d, &b, k);
                assert_eq!(bits(&fa), bits(&ra), "factor, d {d}");
            }
        }
        // Not positive definite: refused by both.
        let mut bad = vec![1.0, 2.0, 2.0, 1.0];
        assert!(cholesky_solve(&mut bad.clone(), 2, &[1.0, 1.0], 1).is_none());
        assert!(reference_solve(&mut bad, 2, &[1.0, 1.0], 1).is_none());
    }

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
