use super::{Location, Signal};
use super::super::types::SignalId;
use super::html::html_escape;

// =============================================================================
// Eval Comparison HTML Rendering
// =============================================================================

/// Comparison between gold (ground truth) and predicted entities.
#[derive(Debug, Clone)]
pub struct EvalComparison {
    /// Document text
    pub text: String,
    /// Gold/ground truth signals
    pub gold: Vec<Signal<Location>>,
    /// Predicted signals
    pub predicted: Vec<Signal<Location>>,
    /// Match results
    pub matches: Vec<EvalMatch>,
}

/// Result of matching a gold or predicted signal.
#[derive(Debug, Clone)]
pub enum EvalMatch {
    /// Exact match: gold and predicted align perfectly.
    Correct {
        /// Gold signal ID
        gold_id: SignalId,
        /// Predicted signal ID
        pred_id: SignalId,
    },
    /// Type mismatch: same span, different label.
    TypeMismatch {
        /// Gold signal ID
        gold_id: SignalId,
        /// Predicted signal ID
        pred_id: SignalId,
        /// Gold label
        gold_label: String,
        /// Predicted label
        pred_label: String,
    },
    /// Boundary error: overlapping but not exact span.
    BoundaryError {
        /// Gold signal ID
        gold_id: SignalId,
        /// Predicted signal ID
        pred_id: SignalId,
        /// Intersection over Union
        iou: f64,
    },
    /// False positive: predicted with no gold match.
    Spurious {
        /// Predicted signal ID
        pred_id: SignalId,
    },
    /// False negative: gold with no prediction.
    Missed {
        /// Gold signal ID
        gold_id: SignalId,
    },
}

impl EvalComparison {
    /// Create a comparison from gold and predicted entities.
    ///
    /// # Example
    ///
    /// ```rust
    /// use anno::core::grounded::{EvalComparison};
    /// use anno::{Signal, Location};
    ///
    /// let text = "Marie Curie won the Nobel Prize.";
    /// let gold = vec![
    ///     Signal::new(0, Location::text(0, 11), "Marie Curie", "PER", 1.0),
    ///     Signal::new(1, Location::text(20, 31), "Nobel Prize", "AWARD", 1.0),
    /// ];
    /// let pred = vec![
    ///     Signal::new(0, Location::text(0, 11), "Marie Curie", "PER", 0.95),
    /// ];
    /// let cmp = EvalComparison::compare(text, gold, pred);
    /// assert_eq!(cmp.matches.len(), 2); // 1 correct, 1 missed
    /// ```
    #[must_use]
    pub fn compare(
        text: &str,
        gold: Vec<Signal<Location>>,
        predicted: Vec<Signal<Location>>,
    ) -> Self {
        let mut matches = Vec::new();
        let mut gold_matched = vec![false; gold.len()];
        let mut pred_matched = vec![false; predicted.len()];

        // First pass: find exact matches and type mismatches
        for (pi, pred) in predicted.iter().enumerate() {
            let pred_offsets = match pred.location.text_offsets() {
                Some(o) => o,
                None => continue,
            };

            for (gi, g) in gold.iter().enumerate() {
                if gold_matched[gi] {
                    continue;
                }
                let gold_offsets = match g.location.text_offsets() {
                    Some(o) => o,
                    None => continue,
                };

                // Exact span match
                if pred_offsets == gold_offsets {
                    if pred.label == g.label {
                        matches.push(EvalMatch::Correct {
                            gold_id: g.id,
                            pred_id: pred.id,
                        });
                    } else {
                        matches.push(EvalMatch::TypeMismatch {
                            gold_id: g.id,
                            pred_id: pred.id,
                            gold_label: g.label.to_string(),
                            pred_label: pred.label.to_string(),
                        });
                    }
                    gold_matched[gi] = true;
                    pred_matched[pi] = true;
                    break;
                }
            }
        }

        // Second pass: find boundary errors (overlapping but not exact)
        for (pi, pred) in predicted.iter().enumerate() {
            if pred_matched[pi] {
                continue;
            }
            let pred_offsets = match pred.location.text_offsets() {
                Some(o) => o,
                None => continue,
            };

            for (gi, g) in gold.iter().enumerate() {
                if gold_matched[gi] {
                    continue;
                }
                let gold_offsets = match g.location.text_offsets() {
                    Some(o) => o,
                    None => continue,
                };

                // Check overlap
                if pred_offsets.0 < gold_offsets.1 && pred_offsets.1 > gold_offsets.0 {
                    let iou = pred.location.iou(&g.location).unwrap_or(0.0);
                    matches.push(EvalMatch::BoundaryError {
                        gold_id: g.id,
                        pred_id: pred.id,
                        iou,
                    });
                    gold_matched[gi] = true;
                    pred_matched[pi] = true;
                    break;
                }
            }
        }

        // Remaining unmatched predictions are spurious
        for (pi, pred) in predicted.iter().enumerate() {
            if !pred_matched[pi] {
                matches.push(EvalMatch::Spurious { pred_id: pred.id });
            }
        }

        // Remaining unmatched gold are missed
        for (gi, g) in gold.iter().enumerate() {
            if !gold_matched[gi] {
                matches.push(EvalMatch::Missed { gold_id: g.id });
            }
        }

        Self {
            text: text.to_string(),
            gold,
            predicted,
            matches,
        }
    }

    /// Count correct matches.
    #[must_use]
    pub fn correct_count(&self) -> usize {
        self.matches
            .iter()
            .filter(|m| matches!(m, EvalMatch::Correct { .. }))
            .count()
    }

    /// Count errors (type mismatch + boundary + spurious + missed).
    #[must_use]
    pub fn error_count(&self) -> usize {
        self.matches.len() - self.correct_count()
    }

    /// Calculate precision.
    #[must_use]
    pub fn precision(&self) -> f64 {
        if self.predicted.is_empty() {
            0.0
        } else {
            self.correct_count() as f64 / self.predicted.len() as f64
        }
    }

    /// Calculate recall.
    #[must_use]
    pub fn recall(&self) -> f64 {
        if self.gold.is_empty() {
            0.0
        } else {
            self.correct_count() as f64 / self.gold.len() as f64
        }
    }

    /// Calculate F1.
    #[must_use]
    pub fn f1(&self) -> f64 {
        let p = self.precision();
        let r = self.recall();
        if p + r > 0.0 {
            2.0 * p * r / (p + r)
        } else {
            0.0
        }
    }
}

/// Render an eval comparison as HTML.
///
/// Shows gold vs predicted side by side with error highlighting.
pub fn render_eval_html(cmp: &EvalComparison) -> String {
    render_eval_html_with_title(cmp, "eval comparison")
}

/// Render an eval comparison as HTML, with a custom title.
///
/// The title is used for both the page `<title>` and the top `<h1>`.
#[must_use]
pub fn render_eval_html_with_title(cmp: &EvalComparison, title: &str) -> String {
    let mut html = String::new();
    let title = html_escape(title);

    html.push_str(
        r#"<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="color-scheme" content="dark light">
"#,
    );
    html.push_str(&format!("<title>{}</title>", title));
    html.push_str(r#"
:root{
  color-scheme: light dark;
  --bg:#0a0a0a;
  --panel-bg:#0d0d0d;
  --text:#b0b0b0;
  --text-strong:#fff;
  --muted:#666;
  --border:#222;
  --border-strong:#333;
  --hover:#111;
  --input-bg:#080808;
  --active:#ddd;
  /* Eval entity colors (dark) */
  --gold-bg:#1a2e1a; --gold-br:#4a8a4a; --gold-tx:#88cc88;
  --pred-bg:#1a1a2e; --pred-br:#4a4a8a; --pred-tx:#8888cc;
  /* Match row borders */
  --m-ok:#4a8a4a;
  --m-type:#8a8a4a;
  --m-bound:#4a8a8a;
  --m-fp:#8a4a4a;
  --m-fn:#8a4a8a;
}
@media (prefers-color-scheme: light){
  :root{
    --bg:#ffffff;
    --panel-bg:#f7f7f7;
    --text:#222;
    --text-strong:#000;
    --muted:#555;
    --border:#d6d6d6;
    --border-strong:#c6c6c6;
    --hover:#f0f0f0;
    --input-bg:#ffffff;
    --active:#000;
    --gold-bg:#e9f7e9; --gold-br:#2f8a2f; --gold-tx:#1f5a1f;
    --pred-bg:#e9e9ff; --pred-br:#6c6cff; --pred-tx:#2b2b7a;
    --m-ok:#2f8a2f;
    --m-type:#8a7a2f;
    --m-bound:#2f7a8a;
    --m-fp:#8a2f2f;
    --m-fn:#6a2f8a;
  }
}
html[data-theme='dark']{
  --bg:#0a0a0a; --panel-bg:#0d0d0d; --text:#b0b0b0; --text-strong:#fff;
  --muted:#666; --border:#222; --border-strong:#333; --hover:#111; --input-bg:#080808; --active:#ddd;
  --gold-bg:#1a2e1a; --gold-br:#4a8a4a; --gold-tx:#88cc88;
  --pred-bg:#1a1a2e; --pred-br:#4a4a8a; --pred-tx:#8888cc;
  --m-ok:#4a8a4a; --m-type:#8a8a4a; --m-bound:#4a8a8a; --m-fp:#8a4a4a; --m-fn:#8a4a8a;
}
html[data-theme='light']{
  --bg:#ffffff; --panel-bg:#f7f7f7; --text:#222; --text-strong:#000;
  --muted:#555; --border:#d6d6d6; --border-strong:#c6c6c6; --hover:#f0f0f0; --input-bg:#ffffff; --active:#000;
  --gold-bg:#e9f7e9; --gold-br:#2f8a2f; --gold-tx:#1f5a1f;
  --pred-bg:#e9e9ff; --pred-br:#6c6cff; --pred-tx:#2b2b7a;
  --m-ok:#2f8a2f; --m-type:#8a7a2f; --m-bound:#2f7a8a; --m-fp:#8a2f2f; --m-fn:#6a2f8a;
}

<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font:12px/1.4 monospace;background:var(--bg);color:var(--text);padding:8px}
h1,h2{color:var(--text-strong);font-weight:normal;border-bottom:1px solid var(--border-strong);padding:4px 0;margin:16px 0 8px}
h1{font-size:14px}h2{font-size:12px}
table{width:100%;border-collapse:collapse;font-size:11px;margin:4px 0}
th,td{padding:4px 8px;text-align:left;border:1px solid var(--border)}
th{background:var(--hover);color:var(--muted);font-weight:normal;text-transform:uppercase;font-size:10px}
tr:hover{background:var(--hover)}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.panel{border:1px solid var(--border);background:var(--panel-bg);padding:8px}
.text-box{background:var(--input-bg);border:1px solid var(--border);padding:8px;white-space:pre-wrap;word-break:break-word;line-height:1.6}
.stats{display:flex;gap:24px;padding:8px 0;border-bottom:1px solid var(--border);margin-bottom:8px}
.stat{text-align:center}.stat-v{font-size:18px;color:var(--text-strong)}.stat-l{font-size:9px;color:var(--muted);text-transform:uppercase}
/* Entities */
.e{padding:1px 2px;border-bottom:2px solid}
.seg{cursor:pointer}
.e-gold{background:var(--gold-bg);border-color:var(--gold-br);color:var(--gold-tx)}
.e-pred{background:var(--pred-bg);border-color:var(--pred-br);color:var(--pred-tx)}
.e-active{outline:1px solid var(--active);outline-offset:1px}
/* Match types */
.correct{background:#1a2e1a;border-color:#4a8a4a}
.type-err{background:#2e2e1a;border-color:#8a8a4a}
.boundary{background:#1a2e2e;border-color:#4a8a8a}
.spurious{background:#2e1a1a;border-color:#8a4a4a}
.missed{background:#2e1a2e;border-color:#8a4a8a}
.match-row.correct{border-left:3px solid var(--m-ok)}
.match-row.type-err{border-left:3px solid var(--m-type)}
.match-row.boundary{border-left:3px solid var(--m-bound)}
.match-row.spurious{border-left:3px solid var(--m-fp)}
.match-row.missed{border-left:3px solid var(--m-fn)}
.match-row.active{outline:1px solid var(--muted)}
.sel{color:var(--muted);margin:6px 0 12px}
.metric{font-size:14px;color:var(--muted)}.metric b{color:var(--text-strong)}
</style>
</head>
<body>
"#);

    // Header (with theme toggle)
    html.push_str(&format!(
        "<div class=\"panel-h\" style=\"justify-content:space-between\"><h1>{}</h1><span class=\"toggle\" id=\"theme-toggle\" title=\"toggle theme (auto → dark → light)\">theme: auto</span></div>",
        title
    ));

    // Metrics bar
    html.push_str("<div class=\"stats\">");
    html.push_str(&format!(
        "<div class=\"stat\"><div class=\"stat-v\">{}</div><div class=\"stat-l\">gold</div></div>",
        cmp.gold.len()
    ));
    html.push_str(&format!(
        "<div class=\"stat\"><div class=\"stat-v\">{}</div><div class=\"stat-l\">predicted</div></div>",
        cmp.predicted.len()
    ));
    html.push_str(&format!(
        "<div class=\"stat\"><div class=\"stat-v\">{}</div><div class=\"stat-l\">correct</div></div>",
        cmp.correct_count()
    ));
    html.push_str(&format!(
        "<div class=\"stat\"><div class=\"stat-v\">{}</div><div class=\"stat-l\">errors</div></div>",
        cmp.error_count()
    ));
    html.push_str(&format!(
        "<div class=\"metric\">P=<b>{:.1}%</b> R=<b>{:.1}%</b> F1=<b>{:.1}%</b></div>",
        cmp.precision() * 100.0,
        cmp.recall() * 100.0,
        cmp.f1() * 100.0
    ));
    html.push_str("</div>");

    // Simple selection readout (helps debugging + browser-based verification)
    html.push_str("<div id=\"selection\" class=\"sel\">click a match row to select spans</div>");

    // Side-by-side text
    html.push_str("<div class=\"grid\">");

    // Gold panel
    html.push_str("<div class=\"panel\"><h2>gold (ground truth)</h2><div class=\"text-box\">");
    let gold_spans: Vec<EvalHtmlSpan> = cmp
        .gold
        .iter()
        .map(|s| {
            let (start, end) = s.location.text_offsets().unwrap_or((0, 0));
            EvalHtmlSpan {
                start,
                end,
                label: s.label.to_string(),
                class: "e-gold",
                id: format!("G{}", s.id),
            }
        })
        .collect();
    html.push_str(&annotate_text_spans(&cmp.text, &gold_spans));
    html.push_str("</div></div>");

    // Predicted panel
    html.push_str("<div class=\"panel\"><h2>predicted</h2><div class=\"text-box\">");
    let pred_spans: Vec<EvalHtmlSpan> = cmp
        .predicted
        .iter()
        .map(|s| {
            let (start, end) = s.location.text_offsets().unwrap_or((0, 0));
            EvalHtmlSpan {
                start,
                end,
                label: s.label.to_string(),
                class: "e-pred",
                id: format!("P{}", s.id),
            }
        })
        .collect();
    html.push_str(&annotate_text_spans(&cmp.text, &pred_spans));
    html.push_str("</div></div>");

    html.push_str("</div>");

    // Match table
    html.push_str("<h2>matches</h2><table>");
    html.push_str("<tr><th>type</th><th>gold</th><th>predicted</th><th>notes</th></tr>");

    for (mi, m) in cmp.matches.iter().enumerate() {
        let (class, mtype, gold_text, pred_text, notes, gid, pid) = match m {
            EvalMatch::Correct { gold_id, pred_id } => {
                let g = cmp.gold.iter().find(|s| s.id == *gold_id);
                let p = cmp.predicted.iter().find(|s| s.id == *pred_id);
                (
                    "correct",
                    "✓",
                    g.map(|s| format!("[{}] {}", s.label, s.surface()))
                        .unwrap_or_default(),
                    p.map(|s| format!("[{}] {}", s.label, s.surface()))
                        .unwrap_or_default(),
                    String::new(),
                    Some(format!("G{}", gold_id)),
                    Some(format!("P{}", pred_id)),
                )
            }
            EvalMatch::TypeMismatch {
                gold_id,
                pred_id,
                gold_label,
                pred_label,
            } => {
                let g = cmp.gold.iter().find(|s| s.id == *gold_id);
                let p = cmp.predicted.iter().find(|s| s.id == *pred_id);
                (
                    "type-err",
                    "type",
                    g.map(|s| format!("[{}] {}", s.label, s.surface()))
                        .unwrap_or_default(),
                    p.map(|s| format!("[{}] {}", s.label, s.surface()))
                        .unwrap_or_default(),
                    format!("{} → {}", gold_label, pred_label),
                    Some(format!("G{}", gold_id)),
                    Some(format!("P{}", pred_id)),
                )
            }
            EvalMatch::BoundaryError {
                gold_id,
                pred_id,
                iou,
            } => {
                let g = cmp.gold.iter().find(|s| s.id == *gold_id);
                let p = cmp.predicted.iter().find(|s| s.id == *pred_id);
                (
                    "boundary",
                    "bound",
                    g.map(|s| format!("[{}] \"{}\"", s.label, s.surface()))
                        .unwrap_or_default(),
                    p.map(|s| format!("[{}] \"{}\"", s.label, s.surface()))
                        .unwrap_or_default(),
                    format!("IoU={:.2}", iou),
                    Some(format!("G{}", gold_id)),
                    Some(format!("P{}", pred_id)),
                )
            }
            EvalMatch::Spurious { pred_id } => {
                let p = cmp.predicted.iter().find(|s| s.id == *pred_id);
                (
                    "spurious",
                    "FP",
                    String::new(),
                    p.map(|s| format!("[{}] {}", s.label, s.surface()))
                        .unwrap_or_default(),
                    "false positive".to_string(),
                    None,
                    Some(format!("P{}", pred_id)),
                )
            }
            EvalMatch::Missed { gold_id } => {
                let g = cmp.gold.iter().find(|s| s.id == *gold_id);
                (
                    "missed",
                    "FN",
                    g.map(|s| format!("[{}] {}", s.label, s.surface()))
                        .unwrap_or_default(),
                    String::new(),
                    "false negative".to_string(),
                    Some(format!("G{}", gold_id)),
                    None,
                )
            }
        };

        let mut data_attrs = String::new();
        if let Some(gid) = gid.as_deref() {
            data_attrs.push_str(&format!(" data-gid=\"{}\"", html_escape(gid)));
        }
        if let Some(pid) = pid.as_deref() {
            data_attrs.push_str(&format!(" data-pid=\"{}\"", html_escape(pid)));
        }

        html.push_str(&format!(
            "<tr id=\"M{mid}\" class=\"match-row {class}\"{attrs}><td><a class=\"match-link\" href=\"#M{mid}\">{mtype}</a></td><td>{gold}</td><td>{pred}</td><td>{notes}</td></tr>",
            mid = mi,
            class = class,
            attrs = data_attrs,
            mtype = html_escape(mtype),
            gold = html_escape(&gold_text),
            pred = html_escape(&pred_text),
            notes = html_escape(&notes)
        ));
    }
    html.push_str("</table>");

    html.push_str(
        r#"<script>
(() => {
  // Theme toggle: auto (prefers-color-scheme) → dark → light.
  const themeBtn = document.getElementById('theme-toggle');
  const themeKey = 'anno-theme';
  const applyTheme = (theme) => {
    const t = theme || 'auto';
    if (t === 'auto') {
      delete document.documentElement.dataset.theme;
    } else {
      document.documentElement.dataset.theme = t;
    }
    if (themeBtn) themeBtn.textContent = `theme: ${t}`;
  };
  const readTheme = () => {
    try { return localStorage.getItem(themeKey) || 'auto'; } catch (_) { return 'auto'; }
  };
  const writeTheme = (t) => {
    try { localStorage.setItem(themeKey, t); } catch (_) { /* ignore */ }
  };
  applyTheme(readTheme());
  if (themeBtn) {
    themeBtn.addEventListener('click', () => {
      const cur = readTheme();
      const next = cur === 'auto' ? 'dark' : (cur === 'dark' ? 'light' : 'auto');
      writeTheme(next);
      applyTheme(next);
    });
  }

  function clearActive() {
    document.querySelectorAll(".e-active").forEach((el) => el.classList.remove("e-active"));
    document.querySelectorAll("tr.match-row.active").forEach((el) => el.classList.remove("active"));
  }

  function findSpanEls(eid) {
    if (!eid) return [];
    // New segmented renderer: one span can be split across multiple elements.
    const els = Array.from(document.querySelectorAll(`span.e[data-eids~='${eid}']`));
    if (els.length) return els;
    // Back-compat: older HTML used a single element id.
    const single = document.getElementById(eid);
    return single ? [single] : [];
  }

  function activate(gid, pid, row) {
    clearActive();
    const gEls = findSpanEls(gid);
    const pEls = findSpanEls(pid);
    const sel = document.getElementById("selection");
    gEls.forEach((el) => el.classList.add("e-active"));
    pEls.forEach((el) => el.classList.add("e-active"));
    if (row) row.classList.add("active");
    if (sel) {
      const parts = [];
      if (gEls.length) {
        const lbl = gEls[0].dataset && gEls[0].dataset.label ? ` [${gEls[0].dataset.label}]` : "";
        parts.push(`gold ${gid}${lbl}`);
      }
      if (pEls.length) {
        const lbl = pEls[0].dataset && pEls[0].dataset.label ? ` [${pEls[0].dataset.label}]` : "";
        parts.push(`pred ${pid}${lbl}`);
      }
      sel.textContent = parts.length ? parts.join("  |  ") : "no selection";
    }
    if (row && row.id) {
      // Keep deep links stable without triggering navigation jump.
      // NOTE: single quotes avoid the Rust raw-string delimiter issue with quote+hash.
      history.replaceState(null, "", '#' + row.id);
    }
    const target = gEls[0] || pEls[0];
    if (target) target.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  document.querySelectorAll("tr.match-row[data-gid], tr.match-row[data-pid]").forEach((tr) => {
    tr.addEventListener("click", () => activate(tr.dataset.gid, tr.dataset.pid, tr));
  });

  document.querySelectorAll("a.match-link").forEach((a) => {
    a.addEventListener("click", (ev) => {
      ev.preventDefault();
      const tr = a.closest("tr.match-row");
      if (!tr) return;
      activate(tr.dataset.gid, tr.dataset.pid, tr);
    });
  });

  // Auto-select a match row if the URL has a deep link (e.g. #M12).
  const hash = (location.hash || "").slice(1);
  if (hash && hash.startsWith("M")) {
    const tr = document.getElementById(hash);
    if (tr && tr.classList && tr.classList.contains("match-row")) {
      activate(tr.dataset.gid, tr.dataset.pid, tr);
    }
  }
})();
</script>"#,
    );

    html.push_str("</body></html>");
    html
}

/// Annotate text with multiple labeled spans (used by eval rendering).
#[derive(Debug, Clone)]
pub(super) struct EvalHtmlSpan {
    pub start: usize,
    pub end: usize,
    pub label: String,
    pub class: &'static str,
    pub id: String,
}

pub(super) fn annotate_text_spans(text: &str, spans: &[EvalHtmlSpan]) -> String {
    let char_count = text.chars().count();
    if char_count == 0 || spans.is_empty() {
        return html_escape(text);
    }

    #[derive(Debug, Clone)]
    struct Meta {
        id: String,
        label: String,
        class: &'static str,
        len: usize,
    }
    #[derive(Debug, Clone)]
    struct Event {
        pos: usize,
        meta_idx: usize,
        delta: i32,
    }

    let mut metas: Vec<Meta> = Vec::with_capacity(spans.len());
    let mut events: Vec<Event> = Vec::new();
    let mut boundaries: Vec<usize> = vec![0, char_count];

    for s in spans {
        let start = s.start.min(char_count);
        let end = s.end.min(char_count);
        if start >= end {
            continue;
        }
        let meta_idx = metas.len();
        metas.push(Meta {
            id: s.id.clone(),
            label: s.label.to_string(),
            class: s.class,
            len: end - start,
        });
        boundaries.push(start);
        boundaries.push(end);
        events.push(Event {
            pos: start,
            meta_idx,
            delta: 1,
        });
        events.push(Event {
            pos: end,
            meta_idx,
            delta: -1,
        });
    }

    if metas.is_empty() {
        return html_escape(text);
    }

    boundaries.sort_unstable();
    boundaries.dedup();
    events.sort_by(|a, b| a.pos.cmp(&b.pos).then_with(|| a.delta.cmp(&b.delta)));

    let mut active_counts: Vec<u32> = vec![0; metas.len()];
    let mut active: Vec<usize> = Vec::new();
    let mut ev_idx = 0usize;
    let mut result = String::new();

    for bi in 0..boundaries.len().saturating_sub(1) {
        let pos = boundaries[bi];
        while ev_idx < events.len() && events[ev_idx].pos == pos {
            let e = &events[ev_idx];
            let idx = e.meta_idx;
            if e.delta < 0 {
                if active_counts[idx] > 0 {
                    active_counts[idx] -= 1;
                    if active_counts[idx] == 0 {
                        active.retain(|&x| x != idx);
                    }
                }
            } else {
                active_counts[idx] += 1;
                if active_counts[idx] == 1 {
                    active.push(idx);
                }
            }
            ev_idx += 1;
        }

        let next = boundaries[bi + 1];
        if next <= pos {
            continue;
        }

        let seg_text: String = text.chars().skip(pos).take(next - pos).collect();
        if active.is_empty() {
            result.push_str(&html_escape(&seg_text));
            continue;
        }

        let primary_idx = active
            .iter()
            .copied()
            .min_by_key(|i| metas[*i].len)
            .unwrap_or(active[0]);
        let primary = &metas[primary_idx];
        let mut eids: Vec<&str> = active.iter().map(|i| metas[*i].id.as_str()).collect();
        eids.sort_unstable();
        let data_eids = eids.join(" ");

        let title = format!(
            "eids=[{}] primary={} [{}..{})",
            data_eids, primary.id, pos, next
        );
        result.push_str(&format!(
            "<span class=\"e seg {class}\" data-eids=\"{eids}\" data-label=\"{label}\" data-start=\"{start}\" data-end=\"{end}\" title=\"{title}\">{text}</span>",
            class = primary.class,
            eids = html_escape(&data_eids),
            label = html_escape(&primary.label),
            start = pos,
            end = next,
            title = html_escape(&title),
            text = html_escape(&seg_text)
        ));
    }

    result
}
