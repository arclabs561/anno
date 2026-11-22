//! Locate cell boundaries in fixed-width tables with an underlined header.
//! Byte offsets always refer to the original document.
use std::{borrow::Cow, sync::LazyLock};

use regex::Regex;

static RULER: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^ *-{2,}(?: {2,}-{2,})+ *$").expect("valid regex"));

fn cell_starts(line: &str, columns: &[usize]) -> Option<Vec<usize>> {
    let line = line.trim_end_matches(['\r', '\n']);
    let mut chars = line.char_indices().peekable();
    let mut position = 0;
    let mut starts = Vec::with_capacity(columns.len());
    for &column in columns {
        while position < column {
            chars.next()?;
            position += 1;
        }
        let &(byte, ch) = chars.peek()?;
        // Require left-aligned cells and a real gap at each declared boundary.
        if ch.is_whitespace() || (!starts.is_empty() && !line[..byte].ends_with("  ")) {
            return None;
        }
        if starts.is_empty() && !line[..byte].bytes().all(|ch| ch == b' ') {
            return None;
        }
        starts.push(byte);
    }
    Some(starts)
}

// Replace a single ASCII separator at each boundary. This preserves byte and
// character offsets and lets the date patterns run once over the entire text.
pub(super) fn with_cell_boundaries(text: &str) -> Cow<'_, str> {
    let mut result = Cow::Borrowed(text);
    if !text.contains("--  ") {
        return result;
    }
    let mut mark = |end: usize| {
        if end > 0 && text.as_bytes()[end - 1].is_ascii_whitespace() {
            result.to_mut().replace_range(end - 1..end, "|");
        }
    };
    let mut mark_row = |offset, len, cells: Vec<usize>| {
        mark(offset);
        for start in cells.into_iter().skip(1) {
            mark(offset + start);
        }
        mark(offset + len);
    };
    let mut lines = text.split_inclusive('\n').peekable();
    let mut previous = "";
    let mut offset = 0;
    let mut columns = Vec::new();
    while let Some(line) = lines.next() {
        let ruler = line.trim_end_matches(['\n', '\r']);
        if RULER.is_match(ruler) {
            columns = ruler
                .match_indices('-')
                .filter_map(|(byte, _)| {
                    (byte == 0 || ruler.as_bytes()[byte - 1] == b' ').then_some(byte)
                })
                .collect();
            // Require both a header and a data row at the advertised columns.
            if let Some(header) = cell_starts(previous, &columns).filter(|_| {
                lines
                    .peek()
                    .is_some_and(|row| cell_starts(row, &columns).is_some())
            }) {
                mark_row(offset - previous.len(), previous.len(), header);
            } else {
                columns.clear();
            }
        } else if !columns.is_empty() {
            if let Some(row) = cell_starts(line, &columns) {
                mark_row(offset, line.len(), row);
            } else {
                columns.clear();
            }
        }
        previous = line;
        offset += line.len();
    }
    result
}
