//! End-to-end demo of the Phase 4 `gliner2_fastino_candle` backend with
//! runtime LoRA adapter swap.
//!
//! Loads `fastino/gliner2-multi-v1`, then walks through the full
//! adapter lifecycle:
//!
//!   1. Run inference on the base model
//!   2. Load adapter A → infer
//!   3. Swap to adapter B (no unload first; load_adapter must replace A)
//!   4. Unload → infer (verify base behavior is restored)
//!
//! Prints per-step entity output and per-(step, step) confidence drift
//! so you can see the adapters actually affect inference.
//!
//! ## Usage
//!
//! 1. Generate two synthetic PEFT adapters (no GPU, no training needed):
//!
//!    ```bash
//!    python scripts/make_synthetic_adapters.py
//!    # produces ./adapter_A/ and ./adapter_B/
//!    ```
//!
//! 2. Run the demo (downloads fastino/gliner2-multi-v1 on first run, ~280 MB):
//!
//!    ```bash
//!    cargo run --release -p anno --features gliner2-fastino-candle \
//!        --example gliner2_candle_lora_demo
//!    ```
//!
//!    Override the adapter paths with env vars if needed:
//!
//!    ```bash
//!    GLINER2_ADAPTER_A=./my_legal_adapter \
//!    GLINER2_ADAPTER_B=./my_medical_adapter \
//!    cargo run --release -p anno --features gliner2-fastino-candle \
//!        --example gliner2_candle_lora_demo
//!    ```
//!
//! Total runtime on CPU: ~3-5 minutes (model load + 4 inference passes).
//! On GPU (with `gliner2-fastino-candle-cuda` feature): ~30 seconds.

#![cfg(feature = "gliner2-fastino-candle")]

use std::path::PathBuf;

use std::error::Error;

use anno::backends::gliner2_fastino_candle::GLiNER2FastinoCandle;
use anno::backends::inference::ZeroShotNER;
use anno::Entity;

const TEXT: &str = "Marie Curie won the Nobel Prize in Physics in 1903. \
     Contact john.smith@example.com or call +1-555-867-5309. \
     Her credit card 4532-1234-5678-9010 is on file with SSN 123-45-6789.";

const TYPES: &[&str] = &[
    "person",
    "award",
    "year",
    "email",
    "phone",
    "credit_card",
    "ssn",
];

fn adapter_path(env_var: &str, default: &str) -> PathBuf {
    std::env::var(env_var)
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(default))
}

fn print_entities(label: &str, ents: &[Entity]) {
    println!("\n=== {label} ({} entities) ===", ents.len());
    for e in ents {
        println!(
            "  [{:>3}..{:>3}]  {:<20}  {:?}  conf={:.4}",
            e.start(),
            e.end(),
            format!("{:?}", e.text),
            e.entity_type,
            e.confidence.value(),
        );
    }
}

fn max_score_diff(a: &[Entity], b: &[Entity]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    let mut a_sorted = a.to_vec();
    let mut b_sorted = b.to_vec();
    a_sorted.sort_by_key(|e| (e.start(), e.end(), e.text.clone()));
    b_sorted.sort_by_key(|e| (e.start(), e.end(), e.text.clone()));
    a_sorted
        .iter()
        .zip(b_sorted.iter())
        .map(|(x, y)| (x.confidence.value() - y.confidence.value()).abs())
        .fold(0.0f64, f64::max)
}

fn main() -> Result<(), Box<dyn Error>> {
    let adapter_a = adapter_path("GLINER2_ADAPTER_A", "./adapter_A");
    let adapter_b = adapter_path("GLINER2_ADAPTER_B", "./adapter_B");

    if !adapter_a.exists() {
        return Err(format!(
            "adapter A not found at {} — generate with: \
             python scripts/make_synthetic_adapters.py",
            adapter_a.display()
        )
        .into());
    }
    if !adapter_b.exists() {
        return Err(format!(
            "adapter B not found at {} — generate with: \
             python scripts/make_synthetic_adapters.py",
            adapter_b.display()
        )
        .into());
    }

    println!("📦 Loading base model: fastino/gliner2-multi-v1");
    println!("    (first run downloads ~280 MB; subsequent runs use HF cache)");
    let mut model = GLiNER2FastinoCandle::from_pretrained("fastino/gliner2-multi-v1")?;
    println!("✅ loaded: {model:?}");

    // ── Step 1: base inference ───────────────────────────────────
    println!("\n🔵 STEP 1 — Base model (no adapter)");
    let base = ZeroShotNER::extract_with_types(&model, TEXT, TYPES, 0.5)?;
    print_entities("base", &base);

    // ── Step 2: load A ───────────────────────────────────────────
    println!("\n🟢 STEP 2 — load_adapter('A', {})", adapter_a.display());
    model.load_adapter("A", &adapter_a)?;
    assert_eq!(model.active_adapter(), Some("A"));
    let after_a = ZeroShotNER::extract_with_types(&model, TEXT, TYPES, 0.5)?;
    print_entities("after A", &after_a);
    println!(
        "\n  Δ(base, A) max_score_diff = {:.6e}",
        max_score_diff(&base, &after_a)
    );

    // ── Step 3: swap to B (no unload) ─────────────────────────────
    println!(
        "\n🟡 STEP 3 — load_adapter('B', {}) [direct swap, no unload]",
        adapter_b.display()
    );
    model.load_adapter("B", &adapter_b)?;
    assert_eq!(model.active_adapter(), Some("B"));
    let after_b = ZeroShotNER::extract_with_types(&model, TEXT, TYPES, 0.5)?;
    print_entities("after B", &after_b);
    println!(
        "\n  Δ(A, B)    max_score_diff = {:.6e}",
        max_score_diff(&after_a, &after_b)
    );
    println!(
        "  Δ(base, B) max_score_diff = {:.6e}",
        max_score_diff(&base, &after_b)
    );

    // ── Step 4: unload ───────────────────────────────────────────
    println!("\n⚪ STEP 4 — unload_adapter()");
    model.unload_adapter()?;
    assert_eq!(model.active_adapter(), None);
    let restored = ZeroShotNER::extract_with_types(&model, TEXT, TYPES, 0.5)?;
    print_entities("restored", &restored);
    let drift = max_score_diff(&base, &restored);
    println!("\n  Δ(base, restored) max_score_diff = {drift:.6e}");

    // ── Sanity assertions ───────────────────────────────────────
    println!("\n══════════════ summary ══════════════");
    let ab_diff = max_score_diff(&after_a, &after_b);
    let base_a_diff = max_score_diff(&base, &after_a);
    let base_b_diff = max_score_diff(&base, &after_b);

    let mut ok = true;
    if drift > 1e-5 {
        eprintln!("❌ unload didn't restore base: drift = {drift}");
        ok = false;
    }
    if ab_diff < 1e-7 {
        eprintln!("❌ adapters A and B produced identical output: swap may be broken");
        ok = false;
    }
    if base_a_diff < 1e-7 {
        eprintln!("❌ adapter A didn't change inference: merge_into_base may be broken");
        ok = false;
    }
    if base_b_diff < 1e-7 {
        eprintln!("❌ adapter B didn't change inference: merge_into_base may be broken");
        ok = false;
    }

    if ok {
        println!("✅ everything works");
        println!("  base ↔ A   : {base_a_diff:.6e}");
        println!("  base ↔ B   : {base_b_diff:.6e}");
        println!("  A ↔ B      : {ab_diff:.6e}");
        println!("  unload drift: {drift:.6e}");
    } else {
        return Err("one or more sanity checks failed (see above)".into());
    }

    Ok(())
}
