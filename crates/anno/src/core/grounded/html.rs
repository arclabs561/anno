use super::{GroundedDocument, Location, Signal};
use super::super::types::{SignalId, TrackId};
use std::collections::HashMap;

/// Generate an HTML visualization of a grounded document.
///
/// Brutalist design: monospace, dense tables, no decoration, raw data.
pub fn render_document_html(doc: &GroundedDocument) -> String {
    let mut html = String::new();
    let stats = doc.stats();

    html.push_str(r#"<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="color-scheme" content="dark light">
<title>grounded::GroundedDocument</title>
<style>
:root{
  /* Allow UA widgets (inputs/scrollbars) to match the theme */
  color-scheme: light dark;
  /* Dark (default) */
  --bg:#0a0a0a;
  --panel-bg:#0d0d0d;
  --text:#b0b0b0;
  --text-strong:#fff;
  --muted:#666;
  --border:#222;
  --border-strong:#333;
  --hover:#111;
  --input-bg:#080808;
  --active:#fff;
  --track-strong:rgba(255,255,255,0.35);
  --track-soft:rgba(255,255,255,0.18);
  /* Entity colors (dark) */
  --per-bg:#1a1a2e; --per-br:#4a4a8a; --per-tx:#8888cc;
  --org-bg:#1a2e1a; --org-br:#4a8a4a; --org-tx:#88cc88;
  --loc-bg:#2e2e1a; --loc-br:#8a8a4a; --loc-tx:#cccc88;
  --mis-bg:#1a1a1a; --mis-br:#4a4a4a; --mis-tx:#999;
  --dat-bg:#2e1a1a; --dat-br:#8a4a4a; --dat-tx:#cc8888;
  --badge-y-bg:#1a2e1a; --badge-y-tx:#4a8a4a; --badge-y-br:#2a4a2a;
  --badge-n-bg:#2e2e1a; --badge-n-tx:#8a8a4a; --badge-n-br:#4a4a2a;
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
    --track-strong:rgba(0,0,0,0.25);
    --track-soft:rgba(0,0,0,0.12);
    /* Entity colors (light) */
    --per-bg:#e9e9ff; --per-br:#6c6cff; --per-tx:#2b2b7a;
    --org-bg:#e9f7e9; --org-br:#2f8a2f; --org-tx:#1f5a1f;
    --loc-bg:#fff7db; --loc-br:#8a7a2f; --loc-tx:#5a4d12;
    --mis-bg:#f2f2f2; --mis-br:#8a8a8a; --mis-tx:#333;
    --dat-bg:#ffe9e9; --dat-br:#8a2f2f; --dat-tx:#5a1f1f;
    --badge-y-bg:#e9f7e9; --badge-y-tx:#1f5a1f; --badge-y-br:#9ad19a;
    --badge-n-bg:#fff7db; --badge-n-tx:#5a4d12; --badge-n-br:#e2d39a;
  }
}
html[data-theme='dark']{
  --bg:#0a0a0a; --panel-bg:#0d0d0d; --text:#b0b0b0; --text-strong:#fff;
  --muted:#666; --border:#222; --border-strong:#333; --hover:#111;
  --input-bg:#080808; --active:#fff;
  --track-strong:rgba(255,255,255,0.35); --track-soft:rgba(255,255,255,0.18);
  --per-bg:#1a1a2e; --per-br:#4a4a8a; --per-tx:#8888cc;
  --org-bg:#1a2e1a; --org-br:#4a8a4a; --org-tx:#88cc88;
  --loc-bg:#2e2e1a; --loc-br:#8a8a4a; --loc-tx:#cccc88;
  --mis-bg:#1a1a1a; --mis-br:#4a4a4a; --mis-tx:#999;
  --dat-bg:#2e1a1a; --dat-br:#8a4a4a; --dat-tx:#cc8888;
  --badge-y-bg:#1a2e1a; --badge-y-tx:#4a8a4a; --badge-y-br:#2a4a2a;
  --badge-n-bg:#2e2e1a; --badge-n-tx:#8a8a4a; --badge-n-br:#4a4a2a;
}
html[data-theme='light']{
  --bg:#ffffff; --panel-bg:#f7f7f7; --text:#222; --text-strong:#000;
  --muted:#555; --border:#d6d6d6; --border-strong:#c6c6c6; --hover:#f0f0f0;
  --input-bg:#ffffff; --active:#000;
  --track-strong:rgba(0,0,0,0.25); --track-soft:rgba(0,0,0,0.12);
  --per-bg:#e9e9ff; --per-br:#6c6cff; --per-tx:#2b2b7a;
  --org-bg:#e9f7e9; --org-br:#2f8a2f; --org-tx:#1f5a1f;
  --loc-bg:#fff7db; --loc-br:#8a7a2f; --loc-tx:#5a4d12;
  --mis-bg:#f2f2f2; --mis-br:#8a8a8a; --mis-tx:#333;
  --dat-bg:#ffe9e9; --dat-br:#8a2f2f; --dat-tx:#5a1f1f;
  --badge-y-bg:#e9f7e9; --badge-y-tx:#1f5a1f; --badge-y-br:#9ad19a;
  --badge-n-bg:#fff7db; --badge-n-tx:#5a4d12; --badge-n-br:#e2d39a;
}

*{box-sizing:border-box;margin:0;padding:0}
body{font:12px/1.4 monospace;background:var(--bg);color:var(--text);padding:8px}
h1,h2,h3{color:var(--text-strong);font-weight:normal;border-bottom:1px solid var(--border-strong);padding:4px 0;margin:16px 0 8px}
h1{font-size:14px}h2{font-size:12px}h3{font-size:11px;color:var(--muted)}
 a{color:inherit}
 a:hover{text-decoration:underline}
table{width:100%;border-collapse:collapse;font-size:11px;margin:4px 0}
th,td{padding:4px 8px;text-align:left;border:1px solid var(--border)}
th{background:var(--hover);color:var(--muted);font-weight:normal;text-transform:uppercase;font-size:10px}
tr:hover{background:var(--hover)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:8px}
.panel{border:1px solid var(--border);background:var(--panel-bg);padding:8px}
.panel-h{display:flex;align-items:center;gap:8px}
.toggle{cursor:pointer;user-select:none;color:var(--muted);border:1px solid var(--border);background:var(--bg);padding:2px 6px;font-size:10px}
.panel-collapsed table,.panel-collapsed .panel-body{display:none}
.toolbar{display:flex;gap:8px;align-items:center;margin:8px 0 0}
.toolbar input{width:100%;max-width:520px;background:var(--input-bg);border:1px solid var(--border);color:var(--text);padding:6px 8px;font:12px monospace}
.muted{color:var(--muted)}
.panel-body{white-space:pre-wrap;word-break:break-word}
.text-box{background:var(--input-bg);border:1px solid var(--border);padding:8px;white-space:pre-wrap;word-break:break-word;line-height:1.6}
.e{padding:1px 2px;border-bottom:1px solid}
.seg{cursor:pointer}
.e-per{background:var(--per-bg);border-color:var(--per-br);color:var(--per-tx)}
.e-org{background:var(--org-bg);border-color:var(--org-br);color:var(--org-tx)}
.e-loc{background:var(--loc-bg);border-color:var(--loc-br);color:var(--loc-tx)}
.e-misc{background:var(--mis-bg);border-color:var(--mis-br);color:var(--mis-tx)}
.e-date{background:var(--dat-bg);border-color:var(--dat-br);color:var(--dat-tx)}
.e-track{box-shadow:inset 0 0 0 1px var(--track-strong)}
.e-track-hover{box-shadow:inset 0 0 0 1px var(--track-soft)}
.e-active{outline:2px solid var(--active);outline-offset:1px}
.conf{color:var(--muted);font-size:10px}
.badge{display:inline-block;padding:1px 4px;font-size:9px;text-transform:uppercase}
.badge-y{background:var(--badge-y-bg);color:var(--badge-y-tx);border:1px solid var(--badge-y-br)}
.badge-n{background:var(--badge-n-bg);color:var(--badge-n-tx);border:1px solid var(--badge-n-br)}
.stats{display:flex;gap:16px;padding:8px 0;border-bottom:1px solid var(--border);margin-bottom:8px}
.stat{text-align:center}.stat-v{font-size:18px;color:var(--text-strong)}.stat-l{font-size:9px;color:var(--muted);text-transform:uppercase}
.id{color:var(--muted);font-size:9px}
.kb{color:var(--muted)}
.arrow{color:var(--muted)}
</style>
</head>
<body>
"#);

    // Header with stats
    html.push_str(&format!(
        r#"<div class="panel-h" style="justify-content:space-between"><h1>doc_id="{}" len={}</h1><span class="toggle" id="theme-toggle" title="toggle theme (auto → dark → light)">theme: auto</span></div>"#,
        html_escape(&doc.id),
        doc.text.len()
    ));

    html.push_str(r#"<div class="stats">"#);
    html.push_str(&format!(
        r#"<div class="stat"><div class="stat-v">{}</div><div class="stat-l">signals</div></div>"#,
        stats.signal_count
    ));
    html.push_str(&format!(
        r#"<div class="stat"><div class="stat-v">{}</div><div class="stat-l">tracks</div></div>"#,
        stats.track_count
    ));
    html.push_str(&format!(r#"<div class="stat"><div class="stat-v">{}</div><div class="stat-l">identities</div></div>"#, stats.identity_count));
    html.push_str(&format!(r#"<div class="stat"><div class="stat-v">{:.2}</div><div class="stat-l">avg_conf</div></div>"#, stats.avg_confidence));
    html.push_str(&format!(
        r#"<div class="stat"><div class="stat-v">{}</div><div class="stat-l">linked</div></div>"#,
        stats.linked_track_count
    ));
    html.push_str(&format!(r#"<div class="stat"><div class="stat-v">{}</div><div class="stat-l">untracked</div></div>"#, stats.untracked_count));
    if stats.iconic_count > 0 || stats.hybrid_count > 0 {
        html.push_str(&format!(r#"<div class="stat"><div class="stat-v">{}/{}/{}</div><div class="stat-l">sym/ico/hyb</div></div>"#,
            stats.symbolic_count, stats.iconic_count, stats.hybrid_count));
    }
    html.push_str(r#"</div>"#);

    // Annotated text
    html.push_str(r#"<h2>text</h2>"#);
    html.push_str(r#"<div class="text-box">"#);
    html.push_str(&annotate_text_html(
        &doc.text,
        doc.signals(),
        &doc.signal_to_track,
    ));
    html.push_str(r#"</div>"#);

    // Selection panel (filled by JS)
    html.push_str(
        r#"<h2>selection</h2><div class="panel" id="selection-panel" role="region" aria-label="selection"><div class="panel-h"><h3>selection</h3><span class="muted" id="selection-hint" role="status" aria-live="polite">click a mention / row to see coref track details</span></div><pre class="panel-body" id="selection-body" role="textbox" aria-readonly="true" aria-label="selection details">—</pre></div>"#,
    );

    // Grid layout for three levels
    html.push_str(r#"<div class="grid">"#);

    // Level 1: Signals table
    html.push_str(r#"<div class="panel" id="panel-signals"><div class="panel-h"><h3>signals (level 1)</h3><span class="toggle" data-toggle="panel-signals">toggle</span></div><div class="toolbar"><input id="signal-filter" type="text" placeholder="filter signals: id / label / surface (e.g. 'PER', 'S12', 'Paris')" /><span class="muted" id="signal-filter-count"></span></div><table id="signals-table">"#);
    html.push_str(r#"<tr><th>id</th><th>span</th><th>surface</th><th>label</th><th>conf</th><th>track</th></tr>"#);
    for signal in doc.signals() {
        let (span, start_opt, end_opt) = if let Some((s, e)) = signal.location.text_offsets() {
            (format!("[{},{})", s, e), Some(s), Some(e))
        } else {
            ("bbox".to_string(), None, None)
        };
        let track_id_num = doc.signal_to_track.get(&signal.id).copied();
        let track_id = track_id_num
            .map(|t| format!("T{}", t))
            .unwrap_or_else(|| "-".to_string());
        let track_attr = track_id_num
            .map(|t| format!(r#" data-track="{}""#, t))
            .unwrap_or_default();
        let offs_attr = match (start_opt, end_opt) {
            (Some(s), Some(e)) => format!(r#" data-start="{}" data-end="{}""#, s, e),
            _ => String::new(),
        };
        let neg = if signal.negated { " NEG" } else { "" };
        html.push_str(&format!(
            r#"<tr data-sid="S{sid}" data-label="{label}" data-surface="{surface}"{track_attr}{offs_attr} data-conf="{conf:.2}"><td class="id"><a href='#S{sid}'>S{sid}</a></td><td>{span}</td><td>{surface}</td><td>{label}{neg}</td><td class="conf">{conf:.2}</td><td class="id">{track}</td></tr>"#,
            sid = signal.id,
            span = span,
            surface = html_escape(&signal.surface),
            label = html_escape(signal.label.as_str()),
            neg = neg,
            conf = signal.confidence.value(),
            track = track_id,
            track_attr = track_attr,
            offs_attr = offs_attr
        ));
    }
    html.push_str(r#"</table></div>"#);

    // Level 2: Tracks table
    html.push_str(r#"<div class="panel" id="panel-tracks"><div class="panel-h"><h3>tracks (level 2)</h3><span class="toggle" data-toggle="panel-tracks">toggle</span></div><table id="tracks-table">"#);
    html.push_str(r#"<tr><th>id</th><th>canonical</th><th>type</th><th>|S|</th><th>signals</th><th>identity</th></tr>"#);
    for track in doc.tracks() {
        let entity_type = track
            .entity_type
            .as_ref()
            .map(|t| t.as_str())
            .unwrap_or("-");
        let signals: Vec<String> = track
            .signals
            .iter()
            .map(|s| format!("S{}", s.signal_id))
            .collect();
        let identity = doc
            .identity_for_track(track.id)
            .map(|i| format!("I{}", i.id))
            .unwrap_or_else(|| "-".to_string());
        let linked_badge = if track.identity_id.is_some() {
            r#"<span class="badge badge-y">y</span>"#
        } else {
            r#"<span class="badge badge-n">n</span>"#
        };
        html.push_str(&format!(
            r#"<tr data-tid="{tid}"><td class="id">T{tid}</td><td>{canonical_surface}</td><td>{etype}</td><td>{n}</td><td class="id">{sigs}</td><td class="id">{ident} {badge}</td></tr>"#,
            tid = track.id,
            canonical_surface = html_escape(&track.canonical_surface),
            etype = html_escape(entity_type),
            n = track.len(),
            sigs = html_escape(&signals.join(" ")),
            ident = identity,
            badge = linked_badge
        ));
    }
    html.push_str(r#"</table></div>"#);

    // Level 3: Identities table
    html.push_str(r#"<div class="panel" id="panel-identities"><div class="panel-h"><h3>identities (level 3)</h3><span class="toggle" data-toggle="panel-identities">toggle</span></div><table>"#);
    html.push_str(r#"<tr><th>id</th><th>name</th><th>type</th><th>kb</th><th>kb_id</th><th>aliases</th></tr>"#);
    for identity in doc.identities() {
        let kb = identity.kb_name.as_deref().unwrap_or("-");
        let kb_id = identity.kb_id.as_deref().unwrap_or("-");
        let entity_type = identity
            .entity_type
            .as_ref()
            .map(|t| t.as_str())
            .unwrap_or("-");
        let aliases = if identity.aliases.is_empty() {
            "-".to_string()
        } else {
            identity.aliases.join(", ")
        };
        html.push_str(&format!(
            r#"<tr><td class="id">I{}</td><td>{}</td><td>{}</td><td class="kb">{}</td><td class="kb">{}</td><td>{}</td></tr>"#,
            identity.id, html_escape(&identity.canonical_name), entity_type, kb, kb_id, html_escape(&aliases)
        ));
    }
    html.push_str(r#"</table></div>"#);

    html.push_str(r#"</div>"#); // end grid

    // Signal-Track-Identity mapping (compact view)
    html.push_str(r#"<h2>hierarchy trace</h2><div class="panel"><table>"#);
    html.push_str(r#"<tr><th>signal</th><th></th><th>track</th><th></th><th>identity</th><th>kb_id</th></tr>"#);
    for signal in doc.signals() {
        let track = doc.track_for_signal(signal.id);
        let identity = doc.identity_for_signal(signal.id);

        let track_str = track
            .map(|t| format!("T{} \"{}\"", t.id, html_escape(&t.canonical_surface)))
            .unwrap_or_else(|| "-".to_string());
        let identity_str = identity
            .map(|i| format!("I{} \"{}\"", i.id, html_escape(&i.canonical_name)))
            .unwrap_or_else(|| "-".to_string());
        let kb_str = identity
            .and_then(|i| i.kb_id.as_ref())
            .map(|s| s.as_str())
            .unwrap_or("-");

        html.push_str(&format!(
            r#"<tr><td>S{} "{}"</td><td class="arrow">→</td><td>{}</td><td class="arrow">→</td><td>{}</td><td class="kb">{}</td></tr>"#,
            signal.id, html_escape(&signal.surface), track_str, identity_str, kb_str
        ));
    }
    html.push_str(r#"</table></div>"#);

    // Minimal JS: click a signal row → highlight that mention in the text box.
    // Also support filtering signals by substring match.
    html.push_str(r#"<script>
(() => {
  // Index signal metadata from the signals table, and map signal/track → text elements.
  const signalMeta = new Map();
  document.querySelectorAll('#signals-table tr[data-sid]').forEach((row) => {
    const sid = row.getAttribute('data-sid');
    if (!sid) return;
    signalMeta.set(sid, {
      sid,
      label: row.getAttribute('data-label') || '',
      surface: row.getAttribute('data-surface') || '',
      conf: row.getAttribute('data-conf') || '',
      start: row.getAttribute('data-start'),
      end: row.getAttribute('data-end'),
      track: row.getAttribute('data-track'),
    });
  });

  const signalEls = new Map();
  const addSignalEl = (sid, el) => {
    if (!sid || !el) return;
    const arr = signalEls.get(sid) || [];
    arr.push(el);
    signalEls.set(sid, arr);
  };
  // Old-style inline spans (non-overlapping renderer).
  document.querySelectorAll('span.e[data-sid]').forEach((el) => {
    addSignalEl(el.getAttribute('data-sid'), el);
  });
  // Segmented spans (overlap/discontinuous-safe renderer).
  document.querySelectorAll('span.seg[data-sids]').forEach((el) => {
    const raw = (el.getAttribute('data-sids') || '').trim();
    if (!raw) return;
    raw.split(/\s+/).filter(Boolean).forEach((sid) => addSignalEl(sid, el));
  });

  const trackEls = new Map();
  for (const [sid, els] of signalEls.entries()) {
    const meta = signalMeta.get(sid);
    const tid = meta ? meta.track : null;
    if (!tid) continue;
    const arr = trackEls.get(tid) || [];
    els.forEach((el) => arr.push(el));
    trackEls.set(tid, arr);
  }

  const selectionBody = document.getElementById('selection-body');
  const selectionHint = document.getElementById('selection-hint');
  const defaultHint = selectionHint ? (selectionHint.textContent || '') : '';
  const setSelection = (text) => {
    if (!selectionBody) return;
    selectionBody.textContent = text;
  };
  const setHint = (text) => {
    if (!selectionHint) return;
    selectionHint.textContent = text || defaultHint;
  };

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

  let activeSignalEls = [];
  let activeSignalRow = null;
  const clearActive = () => {
    if (activeSignalEls && activeSignalEls.length) {
      activeSignalEls.forEach((el) => el.classList.remove('e-active'));
    }
    if (activeSignalRow) activeSignalRow.classList.remove('e-active');
    activeSignalEls = [];
    activeSignalRow = null;
  };

  let activeTrack = null;
  let hoverTrack = null;

  const removeTrackClass = (tid, cls) => {
    if (!tid) return;
    const els = trackEls.get(tid);
    if (!els) return;
    els.forEach((el) => el.classList.remove(cls));
  };

  const addTrackClass = (tid, cls) => {
    if (!tid) return;
    const els = trackEls.get(tid);
    if (!els) return;
    els.forEach((el) => el.classList.add(cls));
  };

  const trackSize = (tid) => {
    const els = tid ? trackEls.get(tid) : null;
    return els ? els.length : 0;
  };

  const getTrackSelectionText = (tid) => {
    if (!tid) return 'track: - (untracked)';
    const row = document.querySelector(`#tracks-table tr[data-tid='${tid}']`);
    if (!row) return `track T${tid}`;
    const cells = row.querySelectorAll('td');
    const canonical = (cells[1]?.textContent || '').trim();
    const etype = (cells[2]?.textContent || '').trim();
    const count = (cells[3]?.textContent || '').trim();
    const sigs = (cells[4]?.textContent || '').trim();
    const lines = [];
    lines.push(`track T${tid} canonical="${canonical}" type="${etype}" mentions=${count}`);
    if (sigs) lines.push(`track signals: ${sigs}`);
    return lines.join('\n');
  };

  const renderTrackSelection = (tid) => setSelection(getTrackSelectionText(tid));

  const renderSignalSelectionBySid = (sid) => {
    const meta = signalMeta.get(sid);
    const label = meta ? (meta.label || '') : '';
    const conf = meta ? (meta.conf || '') : '';
    const start = meta ? meta.start : null;
    const end = meta ? meta.end : null;
    const tid = meta ? meta.track : null;
    const lines = [];
    if (start !== null && end !== null) {
      lines.push(`signal ${sid} label=${label} conf=${conf} span=[${start},${end})`);
    } else {
      lines.push(`signal ${sid} label=${label} conf=${conf}`);
    }
    if (meta && meta.surface) lines.push(`surface: ${meta.surface}`);
    lines.push('');
    lines.push(getTrackSelectionText(tid));
    setSelection(lines.join('\n'));
  };

  const setActiveTrack = (tid) => {
    const next = tid || null;
    if (activeTrack === next) return;
    removeTrackClass(activeTrack, 'e-track');
    activeTrack = next;
    if (activeTrack) addTrackClass(activeTrack, 'e-track');
    if (hoverTrack && activeTrack && hoverTrack === activeTrack) {
      removeTrackClass(hoverTrack, 'e-track-hover');
    }
  };

  const setHoverTrack = (tid) => {
    const next = tid || null;
    if (hoverTrack === next) return;
    removeTrackClass(hoverTrack, 'e-track-hover');
    hoverTrack = next;
    if (!hoverTrack) {
      setHint('');
      return;
    }
    if (activeTrack && hoverTrack === activeTrack) {
      setHint(`selected track T${hoverTrack} (${trackSize(hoverTrack)} mentions)`);
      return;
    }
    addTrackClass(hoverTrack, 'e-track-hover');
    setHint(`hover track T${hoverTrack} (${trackSize(hoverTrack)} mentions)`);
  };

  const emitToParentSpan = (start, end) => {
    try {
      if (!window.parent || window.parent === window) return;
      if (start === null || end === null) return;
      window.parent.postMessage({ type: 'anno:activate-span', start: Number(start), end: Number(end) }, '*');
    } catch (_) {
      // ignore: best-effort bridge for iframe containers
    }
  };

  const activateBySpan = (start, end, emit) => {
    if (start === null || end === null || start === undefined || end === undefined) return;
    // Prefer an exact signal span if present; otherwise fall back to the table row metadata.
    const el = document.querySelector(`span.e[data-sid][data-start='${start}'][data-end='${end}']`);
    if (el) {
      const sid = el.getAttribute('data-sid');
      if (sid) activateSignal(sid, emit);
      return;
    }
    const row = document.querySelector(`#signals-table tr[data-start='${start}'][data-end='${end}']`);
    if (!row) return;
    const sid = row.getAttribute('data-sid');
    if (!sid) return;
    activateSignal(sid, emit);
  };

  const activateSignal = (sid, emit) => {
    clearActive();
    const els = signalEls.get(sid) || [];
    if (!els.length) return;
    els.forEach((el) => el.classList.add('e-active'));
    activeSignalEls = els;
    const row = document.querySelector(`#signals-table tr[data-sid='${sid}']`);
    if (row) {
      row.classList.add('e-active');
      activeSignalRow = row;
    }
    const primaryEl = els[0];
    primaryEl.scrollIntoView({ block: 'center', behavior: 'smooth' });
    const meta = signalMeta.get(sid);
    const tid = meta ? meta.track : primaryEl.getAttribute('data-track');
    setActiveTrack(tid);
    renderSignalSelectionBySid(sid);
    if (emit && meta && meta.start !== null && meta.end !== null) {
      emitToParentSpan(meta.start, meta.end);
    }
  };

  // Table click
  const signalsTable = document.getElementById('signals-table');
  if (signalsTable) {
    signalsTable.addEventListener('click', (ev) => {
      const a = ev.target && ev.target.closest ? ev.target.closest("a[href^='#S']") : null;
      const row = ev.target && ev.target.closest ? ev.target.closest('tr[data-sid]') : null;
      const sid = (a && a.getAttribute('href') ? a.getAttribute('href').slice(1) : null) || (row ? row.getAttribute('data-sid') : null);
      if (!sid) return;
      ev.preventDefault();
      activateSignal(sid, true);
      history.replaceState(null, '', '#' + sid);
    });

    // Hover a signals row → preview track highlight
    signalsTable.addEventListener('mouseover', (ev) => {
      const row = ev.target && ev.target.closest ? ev.target.closest('tr[data-sid]') : null;
      if (!row) return;
      const tid = row.getAttribute('data-track');
      setHoverTrack(tid);
    });
    signalsTable.addEventListener('mouseout', (ev) => {
      const to = ev.relatedTarget;
      if (to && signalsTable.contains(to)) return;
      setHoverTrack(null);
    });
  }

  // Clicking an inline entity should also toggle active highlight.
  const pickPrimarySid = (el) => {
    if (!el) return null;
    const p = el.getAttribute('data-primary');
    if (p) return p;
    const raw = (el.getAttribute('data-sids') || '').trim();
    if (!raw) return null;
    const sids = raw.split(/\s+/).filter(Boolean);
    if (!sids.length) return null;
    // Prefer the shortest mention span from metadata.
    let best = sids[0];
    let bestLen = null;
    for (const sid of sids) {
      const meta = signalMeta.get(sid);
      const s = meta && meta.start !== null ? Number(meta.start) : null;
      const e = meta && meta.end !== null ? Number(meta.end) : null;
      const len = (s !== null && e !== null) ? (e - s) : null;
      if (len === null) continue;
      if (bestLen === null || len < bestLen) {
        best = sid;
        bestLen = len;
      }
    }
    return best;
  };

  document.addEventListener('click', (ev) => {
    const span = ev.target && ev.target.closest ? ev.target.closest('span.e[data-sid]') : null;
    if (span) {
      activateSignal(span.getAttribute('data-sid'), true);
      return;
    }
    const seg = ev.target && ev.target.closest ? ev.target.closest('span.seg[data-sids]') : null;
    if (!seg) return;
    activateSignal(pickPrimarySid(seg), true);
  });

  // Hover an inline entity → preview highlight its track
  document.addEventListener('mouseover', (ev) => {
    const span = ev.target && ev.target.closest ? ev.target.closest('span.e[data-sid]') : null;
    if (span) {
      setHoverTrack(span.getAttribute('data-track'));
      return;
    }
    const seg = ev.target && ev.target.closest ? ev.target.closest('span.seg[data-sids]') : null;
    if (!seg) return;
    const sid = pickPrimarySid(seg);
    const meta = sid ? signalMeta.get(sid) : null;
    setHoverTrack(meta ? meta.track : null);
  });
  document.addEventListener('mouseout', (ev) => {
    const span = ev.target && ev.target.closest ? ev.target.closest('span.e[data-sid]') : null;
    const seg = ev.target && ev.target.closest ? ev.target.closest('span.seg[data-sids]') : null;
    if (!span && !seg) return;
    const to = ev.relatedTarget;
    if (to && to.closest && (to.closest('span.e[data-sid]') || to.closest('span.seg[data-sids]'))) return;
    setHoverTrack(null);
  });

  // Clicking a track row → select track (highlight + details)
  const tracksTable = document.getElementById('tracks-table');
  if (tracksTable) {
    tracksTable.addEventListener('click', (ev) => {
      const row = ev.target && ev.target.closest ? ev.target.closest('tr[data-tid]') : null;
      if (!row) return;
      const tid = row.getAttribute('data-tid');
      setActiveTrack(tid);
      renderTrackSelection(tid);
    });
    tracksTable.addEventListener('mouseover', (ev) => {
      const row = ev.target && ev.target.closest ? ev.target.closest('tr[data-tid]') : null;
      if (!row) return;
      setHoverTrack(row.getAttribute('data-tid'));
    });
    tracksTable.addEventListener('mouseout', (ev) => {
      const to = ev.relatedTarget;
      if (to && tracksTable.contains(to)) return;
      setHoverTrack(null);
    });
  }

  // Filter
  const input = document.getElementById('signal-filter');
  const countEl = document.getElementById('signal-filter-count');
  if (input && signalsTable) {
    const update = () => {
      const q = (input.value || '').trim().toLowerCase();
      let shown = 0;
      const rows = signalsTable.querySelectorAll('tr[data-sid]');
      rows.forEach(row => {
        const sid = (row.getAttribute('data-sid') || '').toLowerCase();
        const label = (row.getAttribute('data-label') || '').toLowerCase();
        const surface = (row.getAttribute('data-surface') || '').toLowerCase();
        const ok = !q || sid.includes(q) || label.includes(q) || surface.includes(q);
        row.style.display = ok ? '' : 'none';
        if (ok) shown += 1;
      });
      if (countEl) countEl.textContent = shown + ' shown';
    };
    input.addEventListener('input', update);
    update();
  }

  // Panel toggles
  document.querySelectorAll('[data-toggle]').forEach(btn => {
    btn.addEventListener('click', () => {
      const id = btn.getAttribute('data-toggle');
      const panel = id ? document.getElementById(id) : null;
      if (!panel) return;
      panel.classList.toggle('panel-collapsed');
    });
  });

  // If URL hash is #S123, focus it.
  const hash = (location.hash || '').slice(1);
  if (hash && hash.startsWith('S')) activateSignal(hash, false);

  // Optional: allow parent pages (e.g., dataset explorers) to sync selection across iframes.
  window.addEventListener('message', (ev) => {
    const data = ev && ev.data ? ev.data : null;
    if (!data || data.type !== 'anno:activate-span') return;
    if (typeof data.start !== 'number' || typeof data.end !== 'number') return;
    activateBySpan(data.start, data.end, false);
  });
})();
</script>"#);

    html.push_str(r#"</body></html>"#);
    html
}

pub(super) fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

pub(super) fn annotate_text_html(
    text: &str,
    signals: &[Signal<Location>],
    signal_to_track: &HashMap<SignalId, TrackId>,
) -> String {
    let char_count = text.chars().count();
    if char_count == 0 {
        return String::new();
    }

    #[derive(Debug, Clone)]
    struct SigMeta {
        sid: String,
        label: String,
        conf: f64,
        track_id: Option<TrackId>,
        covered_len: usize,
    }

    #[derive(Debug, Clone)]
    struct Event {
        pos: usize,
        meta_idx: usize,
        delta: i32, // -1 end, +1 start
    }

    // Collect text segments for each signal (supports discontinuous spans).
    let mut metas: Vec<SigMeta> = Vec::new();
    let mut events: Vec<Event> = Vec::new();
    let mut boundaries: Vec<usize> = vec![0, char_count];

    for s in signals {
        let raw_segments: Vec<(usize, usize)> = match &s.location {
            Location::Text { start, end } => vec![(*start, *end)],
            Location::Discontinuous { segments } => segments.clone(),
        };
        if raw_segments.is_empty() {
            continue;
        }

        let mut cleaned: Vec<(usize, usize)> = Vec::new();
        let mut covered_len = 0usize;
        for (start, end) in raw_segments {
            let start = start.min(char_count);
            let end = end.min(char_count);
            if start >= end {
                continue;
            }
            covered_len = covered_len.saturating_add(end - start);
            cleaned.push((start, end));
        }
        if cleaned.is_empty() {
            continue;
        }

        let meta_idx = metas.len();
        let track_id = signal_to_track.get(&s.id).copied();
        metas.push(SigMeta {
            sid: format!("S{}", s.id),
            label: s.label.to_string(),
            conf: s.confidence.value(),
            track_id,
            covered_len,
        });

        for (start, end) in cleaned {
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
        // Apply all events at this boundary.
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

        // Determine primary (for coloring + click default): shortest covered len, then highest conf.
        let primary_idx = active
            .iter()
            .copied()
            .min_by(|a, b| {
                metas[*a]
                    .covered_len
                    .cmp(&metas[*b].covered_len)
                    .then_with(|| {
                        metas[*b]
                            .conf
                            .partial_cmp(&metas[*a].conf)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
            })
            .unwrap_or(active[0]);
        let primary = &metas[primary_idx];

        let class = match primary.label.to_uppercase().as_str() {
            "PER" | "PERSON" => "e-per",
            "ORG" | "ORGANIZATION" | "COMPANY" => "e-org",
            "LOC" | "LOCATION" | "GPE" => "e-loc",
            "DATE" | "TIME" => "e-date",
            _ => "e-misc",
        };

        let mut sids: Vec<&str> = active.iter().map(|i| metas[*i].sid.as_str()).collect();
        sids.sort_unstable();
        let data_sids = sids.join(" ");

        let mut title = format!(
            "sids=[{}] primary={} [{}..{})",
            data_sids, primary.sid, pos, next
        );
        if let Some(t) = primary.track_id {
            title.push_str(&format!(" track=T{}", t));
        }

        result.push_str(&format!(
            r#"<span class="e seg {class}" data-sids="{sids}" data-start="{start}" data-end="{end}" data-primary="{primary}" title="{title}">{text}</span>"#,
            class = class,
            sids = html_escape(&data_sids),
            start = pos,
            end = next,
            primary = html_escape(&primary.sid),
            title = html_escape(&title),
            text = html_escape(&seg_text),
        ));
    }

    result
}
