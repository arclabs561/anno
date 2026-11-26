//! Saved workflow configuration for CLI commands.

use std::fs;

use clap::ValueEnum;
use serde::Deserialize;

use crate::cli::{
    commands::dataset::{DatasetAction, DatasetArgs},
    parser::{Commands, ModelBackend, OutputFormat},
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

/// Apply only settings with the same meaning on the selected command.
pub fn apply(
    name: &str,
    command: Option<&mut Commands>,
    root: &clap::ArgMatches,
) -> Result<(), String> {
    apply_selected(load(name)?, command, root)
}

fn apply_selected(
    config: SelectedConfig,
    command: Option<&mut Commands>,
    root: &clap::ArgMatches,
) -> Result<(), String> {
    let matches = root.subcommand().map(|(_, m)| m);
    macro_rules! inherit {
        ($args:ident, $($field:ident),+) => {{
            $(inherit(matches, stringify!($field), config.$field, &mut $args.$field);)+
        }};
    }
    match command {
        Some(Commands::Extract(a)) => inherit!(a, model, format, threshold),
        Some(Commands::Debug(a)) => inherit!(a, model, coref, link_kb),
        Some(Commands::Batch(a)) => inherit!(a, model, format, coref, link_kb),
        Some(Commands::Pipeline(a)) => inherit!(a, model, format, coref, link_kb),
        Some(Commands::Enhance(a)) => inherit!(a, format, coref, link_kb),
        Some(Commands::Query(a)) if a.command.is_none() => {
            inherit!(a, format)
        }
        #[cfg(feature = "eval")]
        Some(Commands::CrossDoc(a)) => inherit!(a, model, format),
        Some(Commands::Eval(a)) if a.predictions.is_none() => inherit!(a, model),
        Some(Commands::Dataset(DatasetArgs {
            action: DatasetAction::Eval { model, .. },
        })) => {
            let action = matches.and_then(|m| m.subcommand()).map(|(_, m)| m);
            inherit(action, "model", config.model, model);
        }
        Some(Commands::Export(a)) => inherit!(a, model),
        Some(Commands::Explain(a)) => inherit!(a, model),
        Some(Commands::Privacy(a)) => inherit!(a, model),
        Some(Commands::Singleton(a)) => inherit!(a, model),
        Some(Commands::Domain(a)) => inherit!(a, model),
        Some(Commands::Watch(a)) => inherit!(a, model),
        _ => return Err("This operation has no saved workflow settings. Omit --config.".into()),
    }
    Ok(())
}

fn inherit<T, U: From<T>>(
    matches: Option<&clap::ArgMatches>,
    field: &str,
    saved: Option<T>,
    target: &mut U,
) {
    if matches.and_then(|m| m.value_source(field)) != Some(clap::parser::ValueSource::CommandLine) {
        if let Some(value) = saved {
            *target = value.into();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::parser::Cli;
    use clap::{CommandFactory, FromArgMatches};

    fn configured(args: &[&str]) -> Result<Commands, String> {
        let matches = Cli::command().try_get_matches_from(args).unwrap();
        let mut command = Cli::from_arg_matches(&matches).unwrap().command.unwrap();
        let profile = SelectedConfig {
            model: Some(ModelBackend::Pattern),
            format: Some(OutputFormat::Json),
            threshold: Some(0.95),
            coref: Some(true),
            link_kb: Some(true),
        };
        apply_selected(profile, Some(&mut command), &matches)?;
        Ok(command)
    }

    #[test]
    fn pipeline_preserves_clustering_threshold_and_explicit_false() {
        let Commands::Pipeline(a) = configured(&["anno", "p", "--link-kb=false"]).unwrap() else {
            unreachable!()
        };
        assert!(matches!(a.model, ModelBackend::Pattern));
        assert!(matches!(a.format, OutputFormat::Json));
        assert!(a.coref);
        assert!(!a.link_kb);
        assert_eq!(a.threshold, 0.6);
    }

    #[test]
    fn nested_dataset_model_respects_action_level_override() {
        for explicit in [false, true] {
            let mut args = vec!["anno", "ds", "e"];
            if explicit {
                args.extend(["--model", "stacked"]);
            }
            let Commands::Dataset(DatasetArgs {
                action: DatasetAction::Eval { model, .. },
            }) = configured(&args).unwrap()
            else {
                unreachable!()
            };
            assert_eq!(model.name(), if explicit { "stacked" } else { "pattern" });
        }
    }

    #[test]
    fn non_extraction_operations_reject_a_profile() {
        assert!(configured(&["anno", "info"]).is_err());
        assert!(configured(&["anno", "eval", "--predictions", "saved.json"]).is_err());
    }
}
