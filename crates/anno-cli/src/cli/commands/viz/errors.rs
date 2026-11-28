use super::VizFormat;
use anno::viz::errors::EvalErrors;
use std::io::Read;

pub(super) fn render(
    input: impl Read,
    format: VizFormat,
    title: Option<&str>,
) -> Result<String, String> {
    let mut chart = EvalErrors::from_json(input)?;
    if let Some(title) = title {
        chart = chart.with_title(title)?;
    }
    match format {
        VizFormat::Svg => Ok(chart.render_svg()),
        VizFormat::Html => Ok(chart.render_html()),
        VizFormat::Json => chart.to_json(),
    }
}
