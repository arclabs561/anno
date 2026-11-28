// SPDX-License-Identifier: MIT OR Apache-2.0
// Load this script in an editor displaying the exact source cue strings.
async function resolvePlaybackTargets(cues, receipt, member) {
    const ids = new Set();
    for (const cue of cues) {
        if (typeof cue.id !== "string" || !cue.id.trim() || ids.has(cue.id) || typeof cue.text !== "string") {
            throw new Error("Invalid current cue ID or text");
        }
        ids.add(cue.id);
    }
    const expectedHash = receipt.text_sha256;
    const text = cues.map(cue => cue.text).join("\n");
    const [start, end] = member.chars;
    const scalars = Array.from(text);
    if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) ||
        start < 0 || start >= end || end > scalars.length ||
        scalars.slice(start, end).join("") !== member.quote) {
        throw new Error("The saved quote does not match its transcript range");
    }

    const targets = [];
    let offset = 0;
    for (const cue of cues) {
        const chars = Array.from(cue.text);
        const left = Math.max(0, start - offset);
        const right = Math.min(chars.length, end - offset);
        offset += chars.length + 1; // The producer joins cues with one newline.
        if (left >= right) continue;
        const saved = member.cues[targets.length];
        if (!saved || saved.id !== cue.id || saved.chars[0] !== left || saved.chars[1] !== right) {
            throw new Error("The caption cue layout has changed");
        }
        if (!Number.isSafeInteger(cue.start_ms) || !Number.isSafeInteger(cue.end_ms) ||
            cue.start_ms < 0 || cue.start_ms >= cue.end_ms) {
            throw new Error("Invalid current caption times");
        }
        targets.push({
            id: cue.id,
            startTime: cue.start_ms / 1000,
            endTime: cue.end_ms / 1000,
            // DOM text-node offsets count UTF-16 code units.
            selectionStart: chars.slice(0, left).join("").length,
            selectionEnd: chars.slice(0, right).join("").length,
        });
    }
    if (targets.length !== member.cues.length) throw new Error("The caption cue layout has changed");
    const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
    const hash = Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, "0")).join("");
    if (hash !== expectedHash) {
        throw new Error("The caption transcript has changed");
    }
    return targets;
}
