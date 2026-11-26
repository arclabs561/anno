import Std

/-!
Model of `Mention::{overlaps,len,is_empty}` in crates/anno/src/core/coref.rs.
Offsets are natural numbers. Natural subtraction truncates at zero, as does
usize::saturating_sub. The results therefore apply to all representable offsets.
Run with Lean 4.19.0: lean verification/mention_spans/MentionSpans.lean
-/
namespace Anno.Mention

structure Span where
  start : Nat
  stop : Nat

def overlaps (a b : Span) : Bool :=
  decide (a.start < a.stop ∧ b.start < b.stop ∧ a.start < b.stop ∧ b.start < a.stop)

def len (a : Span) : Nat := a.stop - a.start

def contains (a : Span) (i : Nat) : Prop := a.start ≤ i ∧ i < a.stop

theorem empty_iff (a : Span) : len a = 0 ↔ a.stop ≤ a.start := by
  unfold len
  omega

/-- Exact characterization, including zero-length and reversed spans. -/
theorem shares_character_iff (a b : Span) :
    (∃ i, contains a i ∧ contains b i) ↔
      overlaps a b = true := by
  simp only [overlaps, decide_eq_true_eq, len, contains]
  constructor
  · rintro ⟨i, ⟨ha, ha'⟩, hb, hb'⟩
    omega
  · intro h
    exact ⟨max a.start b.start, by omega⟩

/-- Regression: an empty span inside a nonempty span shares no character. -/
example : overlaps ⟨1, 1⟩ ⟨0, 2⟩ = false ∧ len ⟨1, 1⟩ = 0 := by decide

end Anno.Mention
