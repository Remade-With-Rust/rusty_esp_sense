//! The encoder: a fixed random projection and a ReLU.
//!
//! Nothing here is learned. RuView's MM-Fi study found a random frozen
//! encoder within 2-4 points of a fully trained one (within 2 points across
//! subjects), and the reason is the whole story of channel-state sensing:
//! the representation does not transfer between people, rooms or radios,
//! the readout fitted where the device is does. So the encoder is a seed,
//! its weights regenerated identically on every machine from a generator
//! written here -- not a library's, whose sequence could change under a
//! version bump and silently change every model file.

use candle_core::{Device, Tensor};

use crate::prof::{self, Counter, Stage};

/// `splitmix64`: small, fast, and fixed by its definition.
#[derive(Debug, Clone)]
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `(0, 1)`, never 0.
    fn unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    /// Standard normal (Box-Muller, one of the pair).
    fn normal(&mut self) -> f64 {
        let (u, v) = (self.unit(), self.unit());
        (-2.0 * u.ln()).sqrt() * (core::f64::consts::TAU * v).cos()
    }
}

/// The encoder.
#[derive(Debug, Clone)]
pub struct RandomFeatures {
    /// What generated it.
    pub seed: u64,
    /// Input width.
    pub input: usize,
    /// Output width.
    pub output: usize,
    w: Tensor,
    b: Tensor,
}

impl RandomFeatures {
    /// The encoder for `seed`: weights `N(0, 1/input)`, biases `U(-1, 1)`.
    ///
    /// # Errors
    ///
    /// A tensor error (only on allocation failure).
    pub fn new(seed: u64, input: usize, output: usize) -> crate::Result<Self> {
        let _g = prof::scope(Stage::EncoderGen);
        prof::add(Counter::RandomValues, (input * output + output) as u64);
        let mut g = SplitMix64(seed);
        let scale = 1.0 / (input.max(1) as f64).sqrt();
        let w: Vec<f32> = (0..input * output)
            .map(|_| (g.normal() * scale) as f32)
            .collect();
        let b: Vec<f32> = (0..output).map(|_| (g.unit() * 2.0 - 1.0) as f32).collect();
        Ok(RandomFeatures {
            seed,
            input,
            output,
            w: Tensor::from_vec(w, (output, input), &Device::Cpu)?,
            b: Tensor::from_vec(b, output, &Device::Cpu)?,
        })
    }

    /// `relu(x · Wᵀ + b)` for `x` of shape `[n, input]`.
    ///
    /// # Errors
    ///
    /// A tensor error when `x` is not `[n, input]`.
    pub fn encode(&self, x: &Tensor) -> crate::Result<Tensor> {
        let _g = prof::scope(Stage::Encode);
        prof::add(
            Counter::EncodeMacs,
            (x.dim(0)? * self.input * self.output) as u64,
        );
        // The bias and the ReLU in place over the product: the same f32 add
        // and the same max with zero, per element, where the broadcast add
        // and relu each built another tensor the size of the product.
        let y = x.matmul(&self.w.t()?)?;
        let (n, width) = y.dims2()?;
        let b: Vec<f32> = self.b.to_vec1()?;
        let mut v: Vec<f32> = y.flatten_all()?.to_vec1()?;
        for row in v.chunks_exact_mut(width) {
            for (e, &bias) in row.iter_mut().zip(&b) {
                *e = (*e + bias).max(0.0);
            }
        }
        Ok(Tensor::from_vec(v, (n, width), &Device::Cpu)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_same_seed_is_the_same_encoder_on_every_machine() {
        let a = RandomFeatures::new(7, 5, 3).unwrap();
        let b = RandomFeatures::new(7, 5, 3).unwrap();
        let c = RandomFeatures::new(8, 5, 3).unwrap();
        let v = |e: &RandomFeatures| e.w.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(v(&a), v(&b));
        assert_ne!(v(&a), v(&c));
        // The generator's first output is pinned: a change to it would
        // change every saved model's meaning.
        assert_eq!(SplitMix64(0).next_u64(), 0xE220_A839_7B1D_CDAF);
    }

    #[test]
    fn weights_have_the_stated_scale() {
        let e = RandomFeatures::new(1, 400, 400).unwrap();
        let w = e.w.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let n = w.len() as f64;
        let mean = w.iter().map(|&x| f64::from(x)).sum::<f64>() / n;
        let var = w
            .iter()
            .map(|&x| (f64::from(x) - mean).powi(2))
            .sum::<f64>()
            / n;
        assert!(mean.abs() < 0.002, "{mean}");
        assert!(
            (var * 400.0 - 1.0).abs() < 0.02,
            "variance 1/input: {}",
            var * 400.0
        );
    }

    #[test]
    fn encoding_is_a_relu_of_the_projection() {
        let e = RandomFeatures::new(3, 4, 6).unwrap();
        let x = Tensor::from_vec(vec![1f32, -2.0, 0.5, 3.0], (1, 4), &Device::Cpu).unwrap();
        let y = e.encode(&x).unwrap().to_vec2::<f32>().unwrap();
        assert_eq!(y[0].len(), 6);
        assert!(y[0].iter().all(|&v| v >= 0.0));
    }
}
