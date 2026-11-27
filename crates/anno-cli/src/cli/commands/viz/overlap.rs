use super::VizFormat;
use anno::viz::overlap::Overlap;
use std::io::Read;

pub(super) fn render(
    input: impl Read,
    format: VizFormat,
    title: Option<&str>,
) -> Result<String, String> {
    let mut report = Overlap::from_json(input)?;
    if let Some(title) = title {
        report.set_title(title.to_owned())?;
    }
    match format {
        VizFormat::Svg => report.render_svg(),
        VizFormat::Html => report.render_html(),
        VizFormat::Json => report.to_json(),
    }
}
