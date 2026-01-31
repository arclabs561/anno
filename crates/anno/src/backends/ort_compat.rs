//! ONNX Runtime (ort) compatibility helpers.
//!
//! In multi-crate checkouts, `ort` and `ndarray` versions can drift across crates.
//! To keep backends resilient, we avoid relying on `ort`'s `ndarray`-typed
//! `Tensor::from_array(Array)` impls and instead route through the “(shape, data)”
//! constructor, which is stable and does not depend on matching `ndarray` crate versions.

#[cfg(feature = "onnx")]
use ort::value::Tensor;

#[cfg(feature = "onnx")]
/// Create an `ort::value::Tensor` from an owned ndarray array by extracting `(shape, data)`.
pub fn tensor_from_ndarray<T, D>(
    arr: ndarray::ArrayBase<ndarray::OwnedRepr<T>, D>,
) -> ort::Result<Tensor<T>>
where
    T: ort::tensor::PrimitiveTensorElementType + Clone + std::fmt::Debug + 'static,
    D: ndarray::Dimension,
{
    let shape: Vec<usize> = arr.shape().to_vec();
    let (data, offset) = arr.into_raw_vec_and_offset();
    debug_assert_eq!(offset, Some(0));
    let data = data.into_boxed_slice();
    Tensor::from_array((shape, data))
}
