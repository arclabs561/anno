//! Config command - Configuration management

use clap::{Parser, Subcommand};
use std::fs;

use super::super::output::color;
use super::super::utils::get_config_dir;

/// Configuration management
#[derive(Parser, Debug)]
pub struct ConfigArgs {
    /// Action to perform
    #[command(subcommand)]
    pub action: ConfigAction,
}

#[derive(Subcommand, Debug)]
pub enum ConfigAction {
    /// Save current settings as config
    Save {
        /// Config name
        #[arg(value_name = "NAME")]
        name: String,

        /// Model to save in config
        #[arg(long, value_name = "MODEL")]
        model: Option<String>,

        /// Include coreference in config
        #[arg(long)]
        coref: bool,

        /// Include KB linking in config
        #[arg(long)]
        link_kb: bool,

        /// Threshold for cross-doc
        #[arg(long, value_name = "FLOAT")]
        threshold: Option<f64>,
    },

    /// List saved configs
    #[command(visible_alias = "ls")]
    List,

    /// Show config details
    Show {
        /// Config name
        #[arg(value_name = "NAME")]
        name: String,
    },

    /// Delete config
    Delete {
        /// Config name
        #[arg(value_name = "NAME")]
        name: String,
    },
}

pub fn run(args: ConfigArgs) -> Result<(), String> {
    let config_dir = get_config_dir()?;

    match args.action {
        ConfigAction::Save {
            name,
            model,
            coref,
            link_kb,
            threshold,
        } => {
            #[cfg(all(feature = "cli", feature = "eval"))]
            {
                use toml::Value;

                let mut config = toml::map::Map::new();

                if let Some(ref m) = model {
                    config.insert("model".to_string(), Value::String(m.clone()));
                }
                if coref {
                    config.insert("coref".to_string(), Value::Boolean(true));
                }
                if link_kb {
                    config.insert("link_kb".to_string(), Value::Boolean(true));
                }
                if let Some(t) = threshold {
                    config.insert("threshold".to_string(), Value::Float(t));
                }

                let toml_string = toml::to_string(&config)
                    .map_err(|e| format!("Failed to serialize config: {}", e))?;

                let config_file = config_dir.join(format!("{}.toml", name));
                fs::write(&config_file, toml_string)
                    .map_err(|e| format!("Failed to write config: {}", e))?;

                println!("{} Saved config: {}", color("32", "✓"), name);
            }

            #[cfg(not(all(feature = "cli", feature = "eval")))]
            {
                return Err("Config management requires 'cli' and 'eval' features".to_string());
            }
        }
        ConfigAction::List => {
            if !config_dir.exists() {
                println!("No configs found");
                return Ok(());
            }

            let entries = fs::read_dir(&config_dir)
                .map_err(|e| format!("Failed to read config directory: {}", e))?;

            let mut configs: Vec<_> = entries
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.path()
                        .extension()
                        .and_then(|ext| ext.to_str())
                        .map(|ext| ext == "toml")
                        .unwrap_or(false)
                })
                .map(|e| {
                    e.path()
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .map(|s| s.to_string())
                        .unwrap_or_default()
                })
                .collect();

            configs.sort();

            if configs.is_empty() {
                println!("No configs found");
            } else {
                println!("Saved configs:");
                for config in configs {
                    println!("  {}", config);
                }
            }
        }
        ConfigAction::Show { name } => {
            let config_file = config_dir.join(format!("{}.toml", name));
            if !config_file.exists() {
                return Err(format!("Config '{}' not found", name));
            }

            let content = fs::read_to_string(&config_file)
                .map_err(|e| format!("Failed to read config: {}", e))?;
            println!("Config: {}", name);
            println!("{}", content);
        }
        ConfigAction::Delete { name } => {
            let config_file = config_dir.join(format!("{}.toml", name));
            if !config_file.exists() {
                return Err(format!("Config '{}' not found", name));
            }

            fs::remove_file(&config_file).map_err(|e| format!("Failed to delete config: {}", e))?;
            println!("{} Deleted config: {}", color("32", "✓"), name);
        }
    }

    Ok(())
}
