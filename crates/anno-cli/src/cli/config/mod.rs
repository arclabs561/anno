//! Saved workflow configuration for CLI commands.

use std::fs;

use clap::ValueEnum;
use serde::Deserialize;

use crate::cli::{
    commands::{DebugArgs, ExtractArgs},
    parser::{ModelBackend, OutputFormat},
    utils::{get_config_dir, validate_path_component},
};

/// A validated saved configuration, ready to be merged into command arguments.
#[derive(Debug, Default)]
pub struct SelectedConfig {
    model: Option<ModelBackend>,
    format: Option<OutputFormat>,
    coref: Option<bool>,
    link_kb: Option<bool>,
    threshold: Option<f64>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SavedConfigFile {
    model: Option<String>,
    format: Option<String>,
    coref: Option<bool>,
    link_kb: Option<bool>,
    threshold: Option<f64>,
}

/// Load the saved configuration selected by name.
pub fn load(name: &str) -> Result<SelectedConfig, String> {
    validate_path_component(name, "config name")?;
    load_from_dir(name, &get_config_dir()?)
}

fn load_from_dir(name: &str, config_dir: &std::path::Path) -> Result<SelectedConfig, String> {
    let config_file = config_dir.join(format!("{name}.toml"));
    if !config_file.is_file() {
        return Err(format!("Config '{name}' not found"));
    }
    let contents = fs::read_to_string(&config_file)
        .map_err(|error| format!("Failed to read config '{name}': {error}"))?;
    let saved: SavedConfigFile = toml::from_str(&contents)
        .map_err(|error| format!("Failed to parse config '{name}': {error}"))?;

    Ok(SelectedConfig {
        model: parse_value(name, "model", saved.model)?,
        format: parse_value(name, "format", saved.format)?,
        coref: saved.coref,
        link_kb: saved.link_kb,
        threshold: saved.threshold,
    })
}

fn parse_value<T: ValueEnum + Clone>(
    config_name: &str,
    field: &str,
    value: Option<String>,
) -> Result<Option<T>, String> {
    value
        .map(|value| {
            T::from_str(&value, true).map_err(|_| {
                format!(
                    "Invalid {field} value '{value}' in config '{config_name}'. Run `anno extract --help` to list valid values."
                )
            })
        })
        .transpose()
}

/// Merge a saved config into extract arguments. Explicit command-line values win.
pub fn apply_to_extract(mut args: ExtractArgs) -> Result<ExtractArgs, String> {
    let selected = args.config.as_deref().map(load).transpose()?;
    if let Some(config) = selected {
        args = apply_selected_to_extract(args, config);
    }
    Ok(args)
}

fn apply_selected_to_extract(mut args: ExtractArgs, config: SelectedConfig) -> ExtractArgs {
    args.model = args.model.or(config.model);
    args.format = args.format.or(config.format);
    args.threshold = args.threshold.or(config.threshold);
    args
}

/// Merge a saved config into debug arguments. Explicit command-line flags win when set.
pub fn apply_to_debug(mut args: DebugArgs) -> Result<DebugArgs, String> {
    let selected = args.config.as_deref().map(load).transpose()?;
    if let Some(config) = selected {
        args = apply_selected_to_debug(args, config);
    }
    Ok(args)
}

fn apply_selected_to_debug(mut args: DebugArgs, config: SelectedConfig) -> DebugArgs {
    args.model = args.model.or(config.model);
    args.coref = args.coref.or(config.coref);
    args.link_kb = args.link_kb.or(config.link_kb);
    args
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;
    use clap::Parser;

    #[test]
    fn selected_config_is_typed_and_cli_values_take_precedence() {
        let temp = tempfile::tempdir().unwrap();
        fs::write(
            temp.path().join("workflow.toml"),
            "model = 'pattern'\nformat = 'json'\nthreshold = 0.9\n",
        )
        .unwrap();

        let args = ExtractArgs::try_parse_from([
            "anno",
            "--config",
            "workflow",
            "--model",
            "heuristic",
            "--format",
            "tsv",
            "--threshold",
            "0.4",
            "--text",
            "Alice",
        ])
        .unwrap();
        let args = apply_selected_to_extract(args, load_from_dir("workflow", temp.path()).unwrap());

        assert!(matches!(args.model, Some(ModelBackend::Heuristic)));
        assert!(matches!(args.format, Some(OutputFormat::Tsv)));
        assert_eq!(args.threshold, Some(0.4));
    }

    #[test]
    fn invalid_saved_value_names_its_config_and_field() {
        let temp = tempfile::tempdir().unwrap();
        fs::write(temp.path().join("broken.toml"), "model = 'missing-model'\n").unwrap();

        let error = load_from_dir("broken", temp.path()).unwrap_err();
        assert!(error.contains("model"));
        assert!(error.contains("broken"));
    }

    #[test]
    fn explicit_false_debug_flags_override_saved_true_values() {
        let temp = tempfile::tempdir().unwrap();
        fs::write(
            temp.path().join("debug.toml"),
            "coref = true\nlink_kb = true\n",
        )
        .unwrap();

        let args = DebugArgs::try_parse_from([
            "anno",
            "--config",
            "debug",
            "--coref=false",
            "--link-kb=false",
            "--text",
            "Alice",
        ])
        .unwrap();
        let args = apply_selected_to_debug(args, load_from_dir("debug", temp.path()).unwrap());

        assert_eq!(args.coref, Some(false));
        assert_eq!(args.link_kb, Some(false));
    }
}
