use super::{RenderArgs, VizFormat};
use anno::viz::{Tradeoff, TradeoffAxis};
use clap::{Args, ValueEnum};

#[derive(Debug, Args)]
pub(super) struct TradeoffArgs {
    #[command(flatten)]
    render: RenderArgs,

    /// Override the report's horizontal axis. Suggestions counts all review items.
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
            if let Some(axis) = self.x_axis {
                study.x_axis = Some(match axis {
                    Axis::Unmatched => TradeoffAxis::Unmatched,
                    Axis::Suggestions => TradeoffAxis::Suggestions,
                });
            }
            match format {
                VizFormat::Svg => study.render_svg(),
                VizFormat::Html => study.render_html(),
                VizFormat::Json => study.to_json(),
            }
        })
    }
}
