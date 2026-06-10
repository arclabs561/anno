//! Re-exports of structure-extraction schema types from
//! [`crate::backends::gliner_multitask::schema`].
//!
//! Phase 2 of `gliner2_fastino` consumes the same shape as the GLiNER v1
//! multi-task backend; users can move between backends with a single
//! `use` change. If a future Phase 4 (Candle path) needs different
//! semantics, fork the types here.

pub use crate::backends::gliner_multitask::schema::{
    ExtractedStructure, FieldType, StructureField, StructureTask, StructureValue, TaskSchema,
};
