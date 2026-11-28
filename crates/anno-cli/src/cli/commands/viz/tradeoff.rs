use super::{RenderArgs, VizFormat};
use anno::viz::{Tradeoff, TradeoffAxis};
use clap::{Args, ValueEnum};

#[derive(Debug, Args)]
pub(super) struct TradeoffArgs {
    #[command(flatten)]
    render: RenderArgs,

    /// Horizontal axis for SVG or HTML (default unmatched). Suggestions counts all review items.
    #[arg(long, value_enum)]
    x_axis: Option<Axis>,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Axis {
    Unmatched,
    Suggestions,
}

impl TradeoffArgs {
    pub(super) fn render(&self) -> Result<(), String> {
        self.render.render(|reader, format, title| {
            let mut study = Tradeoff::from_json(reader)?;
            if let Some(title) = title {
                study.title = title.to_owned();
            }
            let axis = match self.x_axis.unwrap_or(Axis::Unmatched) {
                Axis::Unmatched => TradeoffAxis::Unmatched,
                Axis::Suggestions => TradeoffAxis::Suggestions,
            };
            match format {
                VizFormat::Svg => study.render_svg_with_axis(axis),
                VizFormat::Html => study.render_html_with_axis(axis),
                VizFormat::Json if self.x_axis.is_some() => {
                    Err("--x-axis selects a rendered axis. Use SVG or HTML output.".into())
                }
                VizFormat::Json => study.to_json(),
            }
        })
    }
}
