//! Reading a tensor's values where they are.
//!
//! `to_vec1` always copies. A contiguous CPU `f32` tensor already holds its
//! values in one buffer, and a pass that only READS them (statistics, or a
//! transform writing somewhere else) can borrow that buffer instead.

use candle_core::{CpuStorage, InplaceOp1, Layout, Storage, Tensor, WithDType};

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

/// Rewrite a contiguous CPU `f32` tensor's values in place with `f`.
///
/// Only for a tensor the caller alone holds (a product it just computed):
/// every clone of a tensor shares its storage, so the rewrite is seen
/// through all of them.
///
/// # Errors
///
/// A tensor error when `t` is not a contiguous CPU `f32` tensor.
pub(crate) fn rewrite_f32(t: &Tensor, f: impl Fn(&mut [f32])) -> crate::Result<()> {
    struct Rewrite<F>(F);
    impl<F: Fn(&mut [f32])> InplaceOp1 for Rewrite<F> {
        fn name(&self) -> &'static str {
            "rewrite_f32"
        }
        fn cpu_fwd(&self, storage: &mut CpuStorage, layout: &Layout) -> candle_core::Result<()> {
            let (CpuStorage::F32(v), Some((start, end))) = (storage, layout.contiguous_offsets())
            else {
                return Err(candle_core::Error::Msg(
                    "rewrite_f32: not a contiguous f32 tensor".into(),
                ));
            };
            (self.0)(&mut v[start..end]);
            Ok(())
        }
    }
    Ok(t.inplace_op1(&Rewrite(f))?)
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

    #[test]
    fn a_rewrite_changes_the_values_in_place_and_refuses_what_it_cannot() {
        let t = Tensor::from_vec(vec![1.0f32, -2.0, 3.0, -4.0], (2, 2), &Device::Cpu).unwrap();
        rewrite_f32(&t, |v| v.iter_mut().for_each(|x| *x = x.max(0.0))).unwrap();
        assert_eq!(
            with_f32(&t, <[f32]>::to_vec).unwrap(),
            vec![1.0, 0.0, 3.0, 0.0]
        );
        let f = t.to_dtype(candle_core::DType::F64).unwrap();
        assert!(rewrite_f32(&f, |_| {}).is_err());
    }
}
