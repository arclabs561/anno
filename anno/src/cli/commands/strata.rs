//! Strata command - Hierarchical clustering: reveal strata of abstraction

#[cfg(feature = "eval-advanced")]
use anno_strata::HierarchicalLeiden;
use anno_core::GraphDocument;

use super::super::output::color;
use super::super::parser::OutputFormat;

/// Hierarchical clustering: reveal strata of abstraction
#[derive(clap::Parser, Debug)]
pub struct StrataArgs {
    /// Input file containing GraphDocument (JSON format)
    #[arg(short, long, value_name = "FILE")]
    pub input: Option<String>,

    /// Read GraphDocument from stdin (JSON format)
    #[arg(long)]
    pub stdin: bool,

    /// Clustering method to use
    #[arg(short, long, default_value = "leiden")]
    pub method: String,

    /// Resolution parameter for clustering (higher = more, smaller communities)
    #[arg(short, long, default_value = "1.0")]
    pub resolution: f32,

    /// Number of hierarchical levels to compute
    #[arg(short, long, default_value = "3")]
    pub levels: usize,

    /// Output format
    #[arg(short, long, default_value = "json")]
    pub format: OutputFormat,

    /// Output file path (if not specified, prints to stdout)
    #[arg(short = 'o', long)]
    pub output: Option<String>,

    /// Show progress and detailed cluster information
    #[arg(short, long)]
    pub verbose: bool,
}

#[cfg(feature = "eval-advanced")]
pub fn run(args: StrataArgs) -> Result<(), String> {
    // Validate input source
    if args.input.is_none() && !args.stdin {
        return Err("Either --input <FILE> or --stdin must be specified".to_string());
    }

    if args.input.is_some() && args.stdin {
        return Err("Cannot use both --input and --stdin. Choose one.".to_string());
    }

    // Read GraphDocument
    use std::io::Read;
    let json_content = if args.stdin {
        let mut content = String::new();
        std::io::stdin()
            .lock()
            .read_to_string(&mut content)
            .map_err(|e| format!("Failed to read from stdin: {}", e))?;
        content
    } else {
        // Safe: we validated earlier that input is Some when stdin is false
        let input_path = args.input.as_ref()
            .ok_or_else(|| "Internal error: input path should be set when stdin is false".to_string())?;
        std::fs::read_to_string(input_path)
            .map_err(|e| format!("Failed to read input file {}: {}", input_path, e))?
    };

    // Parse GraphDocument
    let graph: GraphDocument = serde_json::from_str(&json_content)
        .map_err(|e| format!("Failed to parse GraphDocument JSON: {}", e))?;

    if args.verbose {
        eprintln!(
            "Loaded graph with {} nodes and {} edges",
            graph.nodes.len(),
            graph.edges.len()
        );
    }

    // Validate method
    if args.method != "leiden" {
        return Err(format!(
            "Unsupported clustering method: '{}'. Currently only 'leiden' is supported.",
            args.method
        ));
    }

    if args.verbose {
        eprintln!(
            "Clustering with method={}, resolution={}, levels={}",
            args.method, args.resolution, args.levels
        );
    }

    // Perform clustering
    let clusterer = HierarchicalLeiden::new()
        .with_resolution(args.resolution)
        .with_levels(args.levels);
    
    let clustered = clusterer
        .cluster(&graph)
        .map_err(|e| format!("Clustering failed: {}", e))?;

    if args.verbose {
        eprintln!("Clustering completed successfully");
    }

    // Format output
    let output = match args.format {
        OutputFormat::Json => {
            serde_json::to_string_pretty(&clustered)
                .map_err(|e| format!("Failed to serialize output: {}", e))?
        }
        OutputFormat::Jsonl => {
            serde_json::to_string(&clustered)
                .map_err(|e| format!("Failed to serialize output: {}", e))?
        }
        OutputFormat::Human => {
            format_human_output(&clustered, args.levels)
        }
        _ => {
            return Err(format!(
                "Format '{:?}' not supported for strata command. Use: json, jsonl, or human.",
                args.format
            ));
        }
    };

    // Write output
    if let Some(output_path) = &args.output {
        std::fs::write(output_path, output)
            .map_err(|e| format!("Failed to write output to {}: {}", output_path, e))?;
        if args.verbose {
            eprintln!("Output written to {}", output_path);
        }
    } else {
        print!("{}", output);
    }

    Ok(())
}

#[cfg(not(feature = "eval-advanced"))]
pub fn run(_args: StrataArgs) -> Result<(), String> {
    Err("Hierarchical clustering requires 'eval-advanced' feature. Build with: cargo build --features eval-advanced".to_string())
}

fn format_human_output(graph: &GraphDocument, levels: usize) -> String {
    let mut output = String::new();

    output.push_str(&format!(
        "{}\n",
        color("1;36", "Hierarchical Clustering Results")
    ));
    output.push_str(&format!(
        "  Nodes: {}\n",
        graph.nodes.len()
    ));
    output.push_str(&format!(
        "  Edges: {}\n",
        graph.edges.len()
    ));
    output.push_str(&format!(
        "  Levels: {}\n\n",
        levels
    ));

    // Group nodes by community at each level
    for level in 0..levels {
        let level_key = format!("level_{}_community", level);
        let mut communities: std::collections::HashMap<u64, Vec<&str>> = std::collections::HashMap::new();

        for node in &graph.nodes {
            if let Some(community_id) = node.properties.get(&level_key) {
                if let Some(community_val) = community_id.as_u64() {
                    communities
                        .entry(community_val)
                        .or_insert_with(Vec::new)
                        .push(&node.name);
                }
            }
        }

        output.push_str(&format!(
            "{} Level {} Communities: {}\n",
            color("1;33", "="),
            level,
            communities.len()
        ));

        let mut sorted_communities: Vec<_> = communities.iter().collect();
        sorted_communities.sort_by_key(|(id, _)| **id);

        for (community_id, nodes) in sorted_communities.iter().take(10) {
            output.push_str(&format!(
                "  Community {}: {} nodes\n",
                community_id,
                nodes.len()
            ));
            if nodes.len() <= 5 {
                for node_label in nodes.iter() {
                    output.push_str(&format!("    - {}\n", node_label));
                }
            } else {
                for node_label in nodes.iter().take(3) {
                    output.push_str(&format!("    - {}\n", node_label));
                }
                output.push_str(&format!("    ... and {} more\n", nodes.len() - 3));
            }
        }

        if communities.len() > 10 {
            output.push_str(&format!(
                "  ... and {} more communities\n",
                communities.len() - 10
            ));
        }

        output.push('\n');
    }

    output
}

