//! CLI module for anno binary
//!
//! This module contains the command-line interface structure, argument parsing,
//! and command routing. Individual command implementations are in the `commands` submodule.

pub mod commands;
pub mod output;
pub mod parser;
pub mod utils;

pub use output::*;
pub use parser::*;
pub use utils::*;
