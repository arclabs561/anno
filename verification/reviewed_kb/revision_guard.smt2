; Bounded model of preview, two intervening operations, and one atomic apply.
; There are two corpus owners, each with a trigger, a peer, and three possible
; catalog bindings. One bit describes whether the peer shares the identity.
; All initial records have named bindings and consistent membership. A live
; Rust revision token is modeled by the pair (owner, revision).
(set-logic ALL)
(declare-datatypes () ((Key A B C)))
(declare-datatypes () ((Corpus (corpus (revision Int) (trigger Key) (peer Key) (member Bool)))))
(declare-datatypes () ((World (world (left Corpus) (right Corpus)))))
(declare-datatypes () ((Operation Read DirectWrite ReviewedWrite TogglePeer BorrowMut)))

(define-fun lookup ((w World) (slot Bool)) Corpus (ite slot (right w) (left w)))
(define-fun replace ((w World) (slot Bool) (c Corpus)) World
  (ite slot (world (left w) c) (world c (right w))))
(define-fun writeBinding ((c Corpus) (target Key)) Corpus
  (corpus (+ 1 (revision c)) target (ite (member c) target (peer c)) (member c)))

; DirectWrite describes an already-consistent binding: writing its current
; value does no work. A successful ReviewedWrite always retires other reviews.
(define-fun step ((w World) (slot Bool) (operation Operation) (target Key)) World
  (let ((c (lookup w slot)))
    (replace w slot
      (match operation
        ((Read c)
         (DirectWrite (ite (= (trigger c) target) c (writeBinding c target)))
         (ReviewedWrite (writeBinding c target))
         (TogglePeer (corpus (+ 1 (revision c)) (trigger c)
                             (ite (member c) (peer c) (trigger c)) (not (member c))))
         (BorrowMut (corpus (+ 1 (revision c)) (trigger c) (peer c) (member c))))))))

(declare-const initial World)
(assert (= (revision (left initial)) 0))
(assert (= (revision (right initial)) 0))
(assert (=> (member (left initial)) (= (trigger (left initial)) (peer (left initial)))))
(assert (=> (member (right initial)) (= (trigger (right initial)) (peer (right initial)))))
(declare-const origin Bool)
(declare-const destination Bool)
(declare-const slot1 Bool)
(declare-const slot2 Bool)
(declare-const op1 Operation)
(declare-const op2 Operation)
(declare-const key1 Key)
(declare-const key2 Key)
(declare-const target Key)
(define-fun saved () Corpus (lookup initial origin))
(define-fun w1 () World (step initial slot1 op1 key1))
(define-fun w2 () World (step w1 slot2 op2 key2))
(define-fun current () Corpus (lookup w2 destination))

; Independent oracle for staleness: inspect operations since preparation,
; without examining either revision. Reads on either owner are harmless.
(define-fun invalidates ((w World) (slot Bool) (operation Operation) (key Key)) Bool
  (and (= origin slot) (distinct operation Read)
       (or (distinct operation DirectWrite) (distinct key (trigger (lookup w slot))))))
(define-fun stale () Bool
  (or (distinct origin destination) (invalidates initial slot1 op1 key1)
      (invalidates w1 slot2 op2 key2)))

(define-fun accepts () Bool (and (= origin destination) (= (revision saved) (revision current))))
(define-fun applyWith ((authorized Bool)) World
  (ite authorized (replace w2 destination (writeBinding current target)) w2))
(define-fun result () World (applyWith accepts))
(define-fun applied () Corpus (lookup result destination))

(echo "stale-rejection-preserves-world")
(push)
(assert stale)
(assert (or accepts (distinct result w2)))
(check-sat)
(pop)

(echo "fresh-acceptance-preserves-frame")
(push)
(assert (not stale))
(assert (or (not accepts)
            (distinct (trigger applied) target)
            (distinct (peer applied) (ite (member saved) target (peer current)))
            (distinct (member applied) (member current))
            (distinct (revision applied) (+ 1 (revision current)))
            (distinct (lookup result (not destination)) (lookup w2 (not destination)))))
(check-sat)
(pop)

; Exhibit an actual change and an out-of-scope peer in a fresh apply.
(echo "fresh-change-with-unaffected-peer")
(push)
(assert (and (not stale) accepts (not (member saved)) (distinct target (trigger saved))
             (distinct result w2) (= (trigger applied) target) (= (peer applied) (peer saved))))
(check-sat)
(pop)

(define-fun aba () Bool
  (and (= origin destination) (= origin slot1) (= origin slot2)
       (= (trigger saved) A) (member saved) (= target C)
       (= op1 DirectWrite) (= key1 B) (= op2 DirectWrite) (= key2 A)))
(echo "changed-and-restored-binding-rejected")
(push)
(assert (and aba stale (not accepts) (= result w2)))
(check-sat)
(pop)

; Three weak implementations must each admit a concrete counterexample.
(define-fun valuesMatch () Bool
  (and (= origin destination) (= (trigger saved) (trigger current))
       (= (peer saved) (peer current)) (= (member saved) (member current))))
(echo "value-comparison-misses-intervening-write")
(push)
(assert (and aba stale valuesMatch (distinct (applyWith valuesMatch) w2)))
(check-sat)
(pop)

(define-fun revisionsMatch () Bool (= (revision saved) (revision current)))
(echo "revision-comparison-misses-foreign-owner")
(push)
(assert (and (= (left initial) (right initial)) (distinct origin destination)
             (= op1 Read) (= op2 Read) revisionsMatch (not accepts)
             (distinct (applyWith revisionsMatch) w2)))
(check-sat)
(pop)

(echo "overbroad-apply-changes-unreported-peer")
(push)
(define-fun writeBoth () Corpus
  (corpus (+ 1 (revision current)) target target (member current)))
(assert (and (not stale) accepts (not (member saved))
             (distinct (peer writeBoth) (peer current))))
(check-sat)
(pop)
