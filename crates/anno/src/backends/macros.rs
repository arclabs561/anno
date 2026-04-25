//! Macros for generating feature-gated backend stubs.
//!
//! When a backend module is compiled but its runtime feature (e.g., `onnx`, `candle`)
//! is disabled, we still need a stub struct + trait impls so that downstream code
//! can reference the type. These stubs return `Error::FeatureNotAvailable` from
//! every method and report `is_available() == false`.
//!
//! The [`define_feature_stub!`] macro generates these stubs from a compact declaration,
//! eliminating copy-paste across backends.

/// Generate a feature-gated stub struct and trait implementations.
///
/// # Usage
///
/// ```rust,ignore
/// define_feature_stub! {
///     /// Optional doc comment for the stub struct.
///     struct GLiNEROnnx;
///     feature = "onnx";
///     name = "GLiNER-ONNX (unavailable)";
///     description = "GLiNER with ONNX Runtime backend - requires 'onnx' feature";
///     error_msg = "GLiNER-ONNX requires the 'onnx' feature";
///     // Optional: extra inherent methods on the stub
///     methods {
///         pub fn model_name(&self) -> &str { "gliner-not-enabled" }
///         pub fn extract(&self, _text: &str, _entity_types: &[&str], _threshold: f32)
///             -> crate::Result<Vec<crate::Entity>>
///         {
///             Err(crate::Error::FeatureNotAvailable(
///                 "GLiNER-ONNX requires the 'onnx' feature".to_string(),
///             ))
///         }
///     }
///     // Optional: extra trait impls
///     impls {
///         ZeroShotNER
///     }
/// }
/// ```
///
/// # Generated Code
///
/// Behind `#[cfg(not(feature = $feature))]`:
/// - `#[derive(Debug)] pub struct $Name;`
/// - `impl $Name { pub fn new(...) -> Result<Self> { Err(...) } }`
/// - `impl Model for $Name { ... }` (extract_entities returns error, is_available = false)
/// - Optional: `impl ZeroShotNER`
#[allow(unused_macros)]
macro_rules! define_feature_stub {
    (
        $(#[$meta:meta])*
        struct $Name:ident;
        feature = $feature:literal;
        name = $name:expr;
        description = $desc:expr;
        error_msg = $err:expr;
        $(
            methods $methods_block:tt
        )?
        $(
            impls {
                $( $trait_tokens:tt )*
            }
        )?
    ) => {
        $(#[$meta])*
        #[cfg(not(feature = $feature))]
        #[derive(Debug)]
        pub struct $Name;

        #[cfg(not(feature = $feature))]
        $crate::backends::macros::_impl_stub_methods!($Name, $feature, $err $(, $methods_block)?);


        #[cfg(not(feature = $feature))]
        impl $crate::Model for $Name {
            fn extract_entities(
                &self,
                _text: &str,
                _language: Option<$crate::Language>,
            ) -> $crate::Result<Vec<$crate::Entity>> {
                Err($crate::Error::FeatureNotAvailable($err.to_string()))
            }

            fn supported_types(&self) -> Vec<$crate::EntityType> {
                vec![]
            }

            fn is_available(&self) -> bool {
                false
            }

            fn name(&self) -> &'static str {
                $name
            }

            fn description(&self) -> &'static str {
                $desc
            }
        }

        $(
            $crate::backends::macros::_impl_stub_traits!($Name, $feature, $err; $( $trait_tokens )*);
        )?
    };
}

/// Internal helper: dispatch each trait name to the appropriate stub impl.
macro_rules! _impl_stub_traits {
    // Base case: no more traits
    ($Name:ident, $feature:literal, $err:expr; ) => {};

    // Consume leading comma separator between traits
    ($Name:ident, $feature:literal, $err:expr; , $($rest:tt)*) => {
        $crate::backends::macros::_impl_stub_traits!($Name, $feature, $err; $($rest)*);
    };

    // ZeroShotNER
    ($Name:ident, $feature:literal, $err:expr; ZeroShotNER $($rest:tt)*) => {
        #[cfg(not(feature = $feature))]
        impl $crate::backends::inference::ZeroShotNER for $Name {
            fn extract_with_types(
                &self,
                _text: &str,
                _entity_types: &[&str],
                _threshold: f32,
            ) -> $crate::Result<Vec<$crate::Entity>> {
                Err($crate::Error::FeatureNotAvailable($err.to_string()))
            }

            fn extract_with_descriptions(
                &self,
                _text: &str,
                _descriptions: &[&str],
                _threshold: f32,
            ) -> $crate::Result<Vec<$crate::Entity>> {
                Err($crate::Error::FeatureNotAvailable($err.to_string()))
            }

            fn default_types(&self) -> &[&'static str] {
                &[]
            }
        }

        $crate::backends::macros::_impl_stub_traits!($Name, $feature, $err; $( $rest )*);
    };

}

/// Internal helper: emit the stub impl block with new() + optional extra methods.
macro_rules! _impl_stub_methods {
    // With extra methods block
    ($Name:ident, $feature:literal, $err:expr, { $($methods_tokens:tt)* }) => {
        impl $Name {
            /// Create a new instance (stub -- requires feature).
            pub fn new(_model_name: &str) -> $crate::Result<Self> {
                Err($crate::Error::FeatureNotAvailable(
                    concat!($err, ". Build with: cargo build --features ", $feature).to_string(),
                ))
            }

            $($methods_tokens)*
        }
    };
    // Without extra methods
    ($Name:ident, $feature:literal, $err:expr) => {
        impl $Name {
            /// Create a new instance (stub -- requires feature).
            pub fn new(_model_name: &str) -> $crate::Result<Self> {
                Err($crate::Error::FeatureNotAvailable(
                    concat!($err, ". Build with: cargo build --features ", $feature).to_string(),
                ))
            }
        }
    };
}

// Re-export macros so they can be used as `crate::backends::macros::define_feature_stub!`
#[allow(unused_imports)]
pub(crate) use _impl_stub_methods;
#[allow(unused_imports)]
pub(crate) use _impl_stub_traits;
#[allow(unused_imports)]
pub(crate) use define_feature_stub;
