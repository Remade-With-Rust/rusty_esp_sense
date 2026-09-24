//! Reading a tensor's values where they are.
//!
//! `to_vec1` always copies. A contiguous CPU `f32` tensor already holds its
//! values in one buffer, and a pass that only READS them (statistics, or a
//! transform writing somewhere else) can borrow that buffer instead.

use candle_core::{Storage, Tensor, WithDType};

/// Call `f` with `t`'s values as `D`, row-major: borrowed when `t` is a
/// contiguous CPU tensor of that type, from a copy otherwise.
///
/// # Errors
///
/// A tensor error from the fallback copy.
pub(crate) fn with_values<D: WithDType, T>(
    t: &Tensor,
    f: impl FnOnce(&[D]) -> T,
) -> crate::Result<T> {
    if t.dtype() == D::DTYPE {
        let (storage, layout) = t.storage_and_layout();
        if let (Storage::Cpu(cpu), Some((start, end))) = (&*storage, layout.contiguous_offsets()) {
            if let Some(values) = cpu.as_slice::<D>().ok().and_then(|v| v.get(start..end)) {
                return Ok(f(values));
            }
        }
    }
    let v: Vec<D> = t.to_dtype(D::DTYPE)?.flatten_all()?.to_vec1()?;
    Ok(f(&v))
}

/// [`with_values`] for `f32`.
///
/// # Errors
///
/// A tensor error from the fallback copy.
pub(crate) fn with_f32<T>(t: &Tensor, f: impl FnOnce(&[f32]) -> T) -> crate::Result<T> {
    with_values(t, f)
}

#[cfg(test)]
mod tests {
    use candle_core::Device;

    use super::*;

    #[test]
    fn borrowed_and_copied_values_agree() {
        let t = Tensor::from_vec(
            (0..12).map(|i| i as f32).collect::<Vec<_>>(),
            (3, 4),
            &Device::Cpu,
        )
        .unwrap();
        let direct = with_f32(&t, <[f32]>::to_vec).unwrap();
        // A transposed view is not contiguous: the copying path.
        let tt = t.t().unwrap();
        let viewed = with_f32(&tt, <[f32]>::to_vec).unwrap();
        let want: Vec<f32> = tt
            .contiguous()
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(direct, (0..12).map(|i| i as f32).collect::<Vec<_>>());
        assert_eq!(viewed, want);
        // A narrowed view is contiguous at an offset: borrowed, from there.
        let n = t.narrow(0, 1, 2).unwrap();
        assert_eq!(
            with_f32(&n, <[f32]>::to_vec).unwrap(),
            (4..12).map(|i| i as f32).collect::<Vec<_>>()
        );
    }
}
