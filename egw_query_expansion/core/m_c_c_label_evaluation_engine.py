# mcc.py
# Monotone Compliance Contract (MCC) evaluator
# - Stratified Horn-rule engine with monotone aggregates
# - Least fixpoint computation (immediate consequence operator)
# - Monotonicity: adding supportive evidence does not lower labels
# - Explicit mandatory violation handling and contradiction detection
#
# Python 3.12, stdlib only.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

Entity = Any
Predicate = str
Fact = Tuple[Predicate, Tuple[Any, ...]]  # ('p', (x,y,...))
GroundTuple = Tuple[Any, ...]
Label = str

# ----------------------------
# Label Poset (stratification)
# ----------------------------


@dataclass(frozen=True)
class LabelPoset:
    """
    Partially ordered set of labels. Provides:
      - leq(a, b): a <= b
      - covers: immediate predecessors/successors for documentation
      - A topological layering for stratified evaluation.
    """

    labels: FrozenSet[Label]
    order: FrozenSet[
        Tuple[Label, Label]
    ]  # reflexive-transitive closure or at least reflexive + transitive

    def __post_init__(self):
        # Basic checks: reflexivity required for all labels; antisymmetry implied by partial order property (trusted).
        for l in self.labels:
            if (l, l) not in self.order:
                raise ValueError(f"Poset must be reflexive: missing ({l},{l}).")

    def leq(self, a: Label, b: Label) -> bool:
        return (a, b) in self.order

    def lt(self, a: Label, b: Label) -> bool:
        return self.leq(a, b) and a != b

    def max_label(self, labels: Iterable[Label]) -> Label:
        """
        Return a maximal element among labels under the poset. If multiple incomparable maxima exist,
        this raises (design uses stratified chains; rules should ensure comparability for a given entity).
        """
        cand = list(labels)
        if not cand:
            raise ValueError("Empty label set.")
        # pick an element that is not less than any other
        for x in cand:
            if all(self.leq(y, x) for y in cand):
                return x
        raise ValueError("No unique maximal label (incomparable maxima detected).")

    def topological_layers(self) -> List[Set[Label]]:
        """
        Compute a simple layering consistent with the partial order.
        Kahn-like layering on Hasse approximation using order info.
        """
        # Build predecessor map
        preds: Dict[Label, Set[Label]] = {l: set() for l in self.labels}
        for a, b in self.order:
            if a != b:
                preds[b].add(a)

        layers: List[Set[Label]] = []
        remaining = set(self.labels)
        while remaining:
            layer = {l for l in remaining if not (preds[l] & remaining)}
            if not layer:
                # cycle or incomplete order; for MCC we require stratification without cycles
                raise ValueError("Labels are not stratifiable (cycle detected).")
            layers.append(layer)
            remaining -= layer
        return layers


# ----------------------------
# Knowledge base
# ----------------------------


@dataclass
class KnowledgeBase:
    """
    Extensional facts storage (E) + internal derived facts (I) for labels and violations.
    """

    facts: Dict[Predicate, Set[GroundTuple]] = field(default_factory=dict)

    def add_fact(self, pred: Predicate, *args: Any) -> None:
        self.facts.setdefault(pred, set()).add(tuple(args))

    def has_fact(self, pred: Predicate, args: GroundTuple) -> bool:
        return args in self.facts.get(pred, set())

    def get(self, pred: Predicate) -> Set[GroundTuple]:
        return self.facts.get(pred, set())

    def clone(self) -> KnowledgeBase:
        return KnowledgeBase({p: set(v) for p, v in self.facts.items()})


# ----------------------------
# Literals and Aggregates
# ----------------------------


@dataclass(frozen=True)
class Literal:
    """
    Positive literal p(entity, c1, c2, ...) where entity is always the first argument.
    We intentionally avoid arbitrary variable binding to keep evaluation tractable and stratified.
    """

    predicate: Predicate
    const_args: Tuple[Any, ...] = field(default_factory=tuple)

    def ground(self, entity: Entity) -> Tuple[Predicate, GroundTuple]:
        return (self.predicate, (entity,) + self.const_args)


@dataclass(frozen=True)
class MandatoryLiteral(Literal):
    """
    Mandatory literal that must be present; if it is explicitly violated, label transitions may be revoked/demoted.
    Violation is represented by violation(entity, predicate, const_args).
    """

    pass


@dataclass(frozen=True)
class Aggregate:
    """
    Monotone aggregate A over facts affecting an entity. Must be monotone in the sense:
      - adding more base facts cannot reduce the aggregate's truth (once threshold passed, it remains passed).
    We implement common monotone forms as callables over KB returning (holds: bool, value: Any).
    """

    name: str
    func: Callable[[KnowledgeBase, Entity], Tuple[bool, Any]]
    description: str = ""


# ----------------------------
# Rules
# ----------------------------

RuleId = str


@dataclass(frozen=True)
class Rule:
    """
    Horn-like rule:
      head: (entity -> target label)
      body: list of positive literals (supportive)
      mandatory: list of mandatory literals that must be present; their explicit violations can force demotion.
      aggs: list of monotone aggregates; all must hold to fire.
    """

    rule_id: RuleId
    target_label: Label
    body: Tuple[Literal, ...] = field(default_factory=tuple)
    mandatory: Tuple[MandatoryLiteral, ...] = field(default_factory=tuple)
    aggs: Tuple[Aggregate, ...] = field(default_factory=tuple)
    stratum: int = (
        0  # computed from label poset layering; used for stratified evaluation
    )


# ----------------------------
# Derivations and Results
# ----------------------------


@dataclass
class Derivation:
    entity: Entity
    from_label: Label
    to_label: Label
    rule_id: RuleId
    supporting_facts: List[Fact]
    mandatory_facts: List[Fact]
    aggregates: List[Tuple[str, bool, Any, str]]  # (name, holds, value, description)


@dataclass
class EvaluationResult:
    entity_labels: Dict[Entity, Label]
    derivations: List[Derivation]
    contradictions: List[Tuple[Entity, str]]  # (entity, description)


# ----------------------------
# Contradiction detection
# ----------------------------


@dataclass(frozen=True)
class MutualExclusion:
    """
    If both literals are present for the same entity, it's a contradiction.
    """

    a: Literal
    b: Literal
    description: str


# ----------------------------
# Engine
# ----------------------------


class MCCEngine:
    """
    Monotone Compliance Contract evaluator.
    - Unique model via stratification on label poset.
    - T_P iteration until least fixpoint (labels only rise, except explicit mandatory violations).
    - Monotone aggregates ensure monotonicity of support.
    - Contradictions recorded; explicit violations can demote to a designated label.
    """

    def __init__(
        self,
        poset: LabelPoset,
        rules: Sequence[Rule],
        bottom_label: Label,
        violation_label: Optional[Label] = None,
        mutual_exclusions: Sequence[MutualExclusion] = (),
        violation_predicate: Predicate = "violation",
        label_predicate: Predicate = "label",
    ):
        if bottom_label not in poset.labels:
            raise ValueError("bottom_label must be in the poset.")
        if violation_label is not None and violation_label not in poset.labels:
            raise ValueError("violation_label must be in the poset.")

        self.poset = poset
        self.bottom = bottom_label
        self.violation_label = violation_label
        self.mutual_exclusions = tuple(mutual_exclusions)
        self.VIOL = violation_predicate
        self.LABEL = label_predicate

        # Assign strata to rules based on target_label layer in poset
        layer_map: Dict[Label, int] = {}
        for depth, layer in enumerate(self.poset.topological_layers()):
            for lab in layer:
                layer_map[lab] = depth
        self.rules = tuple(
            Rule(
                rule_id=r.rule_id,
                target_label=r.target_label,
                body=r.body,
                mandatory=r.mandatory,
                aggs=r.aggs,
                stratum=layer_map[r.target_label],
            )
            for r in rules
        )

    # ------------------------
    # Aggregates helpers
    # ------------------------

    @staticmethod
    def agg_count_at_least(
        pred: Predicate, k: int, const_filter: Optional[Tuple[int, Any]] = None
    ) -> Aggregate:
        """
        Monotone aggregate: count of facts pred(entity, ...) >= k.
        const_filter: optional (arg_index>0, value) to filter by a constant on that arg position.
        """

        def f(kb: KnowledgeBase, entity: Entity) -> Tuple[bool, Any]:
            cnt = 0
            for args in kb.get(pred):
                if not args:
                    continue
                if args[0] != entity:
                    continue
                if const_filter is not None:
                    idx, val = const_filter
                    if len(args) <= idx:
                        continue
                    if args[idx] != val:
                        continue
                cnt += 1
            return (cnt >= k, cnt)

        desc = f"count {pred}(entity, ...) >= {k}" + (
            f" with filter arg[{const_filter[0]}]=={const_filter[1]}"
            if const_filter
            else ""
        )
        return Aggregate(name=f"count_{pred}_ge_{k}", func=f, description=desc)

    @staticmethod
    def agg_sum_at_least(
        pred: Predicate, arg_index: int, threshold: float
    ) -> Aggregate:
        """
        Monotone aggregate: sum of numeric args at index >= threshold.
        """

        def f(kb: KnowledgeBase, entity: Entity) -> Tuple[bool, Any]:
            s = 0.0
            for args in kb.get(pred):
                if args and args[0] == entity and len(args) > arg_index:
                    v = args[arg_index]
                    try:
                        s += float(v)
                    except Exception:
                        continue
            return (s >= threshold, s)

        desc = f"sum {pred}(entity, *).arg[{arg_index}] >= {threshold}"
        return Aggregate(name=f"sum_{pred}_ge_{threshold}", func=f, description=desc)

    # ------------------------
    # Evaluation
    # ------------------------

    def evaluate(
        self,
        kb: KnowledgeBase,
        entities: Iterable[Entity],
        initial_labels: Optional[Dict[Entity, Label]] = None,
    ) -> EvaluationResult:
        """
        Compute least fixpoint model:
          - Start from bottom label per entity (or provided initial_labels)
          - Iteratively apply rules in increasing stratum order (stratified)
          - Labels only increase under poset unless explicit violation present
          - Detect contradictions; if violation_label is set, demote entity to it
        """
        entities = list(entities)
        state: Dict[Entity, Label] = {e: self.bottom for e in entities}
        if initial_labels:
            for e, lab in initial_labels.items():
                if e in state:
                    if not self.poset.leq(state[e], lab):
                        # Initial labels must be >= bottom in poset
                        raise ValueError("Initial label not >= bottom for entity.")
                    state[e] = lab

        derivations: List[Derivation] = []
        contradictions: List[Tuple[Entity, str]] = []

        # Precompute rule buckets by stratum
        by_stratum: Dict[int, List[Rule]] = {}
        for r in self.rules:
            by_stratum.setdefault(r.stratum, []).append(r)
        strata = sorted(by_stratum.keys())

        changed = True
        while changed:
            changed = False

            # Detect contradictions at each iteration (monotone w.r.t evidence growth)
            for e in entities:
                for mex in self.mutual_exclusions:
                    a_pred, a_args = mex.a.ground(e)
                    b_pred, b_args = mex.b.ground(e)
                    if kb.has_fact(a_pred, a_args) and kb.has_fact(b_pred, b_args):
                        msg = f"Mutually exclusive evidence: {mex.description}"
                        contradictions.append((e, msg))
                        # Record violation fact
                        kb.add_fact(self.VIOL, e, "mutual_exclusion", mex.description)

            # Apply stratified T_P iteration: within each stratum, we may need to loop
            for s in strata:
                rules = by_stratum[s]
                local_progress = True
                while local_progress:
                    local_progress = False

                    for r in rules:
                        for e in entities:
                            cur = state[e]

                            # If explicit violation fact exists and violation_label configured, enforce demotion once.
                            if self.violation_label is not None:
                                if any(v[0] == e for v in kb.get(self.VIOL)):
                                    if self.poset.lt(self.violation_label, cur):
                                        # violation_label is below current -> demotion allowed only by explicit violation
                                        state[e] = self.violation_label
                                        # We do not record derivation for demotion as a Horn derivation; but we could log.
                                        local_progress = True
                                        changed = True
                                        continue

                            # Check mandatory literals presence
                            mandatory_facts: List[Fact] = []
                            violated_mandatory = False
                            for m in r.mandatory:
                                pred, args = m.ground(e)
                                present = kb.has_fact(pred, args)
                                mandatory_facts.append((pred, args))
                                if not present:
                                    violated_mandatory = True
                                    # If absent, rule cannot fire; but absence does not cause demotion (explicit violation handles that)
                                    break
                                # Check explicit violation facts for this mandatory requirement
                                if kb.has_fact(
                                    self.VIOL,
                                    (e, pred, args[1:] if len(args) > 1 else ()),
                                ) or kb.has_fact(self.VIOL, (e, pred)):
                                    violated_mandatory = True
                                    break
                            if violated_mandatory:
                                continue

                            # Check supportive body literals
                            supporting_facts: List[Fact] = []
                            body_ok = True
                            for lit in r.body:
                                pred, args = lit.ground(e)
                                if not kb.has_fact(pred, args):
                                    body_ok = False
                                    break
                                supporting_facts.append((pred, args))
                            if not body_ok:
                                continue

                            # Check aggregates (monotone)
                            agg_results: List[Tuple[str, bool, Any, str]] = []
                            aggs_ok = True
                            for agg in r.aggs:
                                holds, value = agg.func(kb, e)
                                agg_results.append(
                                    (agg.name, holds, value, agg.description)
                                )
                                if not holds:
                                    aggs_ok = False
                                    break
                            if not aggs_ok:
                                continue

                            # Rule fires: propose upgrade to target_label
                            tgt = r.target_label
                            if self.poset.lt(cur, tgt):
                                state[e] = tgt
                                derivations.append(
                                    Derivation(
                                        entity=e,
                                        from_label=cur,
                                        to_label=tgt,
                                        rule_id=r.rule_id,
                                        supporting_facts=supporting_facts,
                                        mandatory_facts=mandatory_facts,
                                        aggregates=agg_results,
                                    )
                                )
                                local_progress = True
                                changed = True
                            # If tgt <= cur, no change (monotone: never demote here)

        return EvaluationResult(
            entity_labels=state, derivations=derivations, contradictions=contradictions
        )


# ----------------------------
# Convenience builders
# ----------------------------


def make_chain_poset(chain: Sequence[Label]) -> LabelPoset:
    """
    Build a total order chain poset: chain[0] <= chain[1] <= ...
    Useful when labels are naturally tiered.
    """
    labels: Set[Label] = set(chain)
    order: Set[Tuple[Label, Label]] = set()
    for i, a in enumerate(chain):
        for j, b in enumerate(chain):
            if i <= j:
                order.add((a, b))
    # add reflexivity in case chain had duplicates filtered
    for l in labels:
        order.add((l, l))
    return LabelPoset(labels=frozenset(labels), order=frozenset(order))


# ----------------------------
# Example usage (self-test)
# ----------------------------

if __name__ == "__main__":
    # Define a simple 4-level compliance chain
    poset = make_chain_poset(["non_compliant", "baseline", "enhanced", "certified"])

    # Build facts
    kb = KnowledgeBase()
    # Example predicates:
    # evidence(entity, type)
    # control(entity, name)
    # metric(entity, "score", value)
    # obligation(entity, name)  -> mandatory
    # violation(entity, ...)
    #
    # Entity E1 has various supportive facts:
    E1 = "E1"
    kb.add_fact("obligation", E1, "data_retention_policy")
    kb.add_fact("evidence", E1, "audit_log_enabled")
    kb.add_fact("control", E1, "mfa")
    kb.add_fact("control", E1, "encryption_at_rest")
    kb.add_fact("metric", E1, "coverage", 0.82)
    kb.add_fact("metric", E1, "coverage", 0.15)  # sum >= 0.97

    # Mutual exclusion: cannot be both 'encryption_at_rest' and 'no_encryption_at_rest' in this toy example
    mex = MutualExclusion(
        a=Literal("control", ("encryption_at_rest",)),
        b=Literal("control", ("no_encryption_at_rest",)),
        description="Encryption and No-Encryption are mutually exclusive",
    )

    # Literals
    L_ev = Literal("evidence", ("audit_log_enabled",))
    L_mfa = Literal("control", ("mfa",))
    M_obl = MandatoryLiteral("obligation", ("data_retention_policy",))

    # Aggregates
    A_cov = MCCEngine.agg_sum_at_least("metric", arg_index=2, threshold=0.95)
    A_ctrls = MCCEngine.agg_count_at_least("control", k=2)

    # Rules:
    # R1: baseline if audit logging AND obligation present
    R1 = Rule(
        rule_id="R1",
        target_label="baseline",
        body=(L_ev,),
        mandatory=(M_obl,),
        aggs=(),
    )
    # R2: enhanced if baseline supports 2+ controls and coverage >= 0.95
    # We model dependency via stratification on target labels; body only contains EDB literals.
    R2 = Rule(
        rule_id="R2",
        target_label="enhanced",
        body=(L_mfa,),
        mandatory=(M_obl,),
        aggs=(A_cov, A_ctrls),
    )
    # R3: certified if enhanced evidence is sufficiently strong (toy extra condition)
    R3 = Rule(
        rule_id="R3",
        target_label="certified",
        body=(Literal("control", ("encryption_at_rest",)),),
        mandatory=(M_obl,),
        aggs=(A_cov,),
    )

    engine = MCCEngine(
        poset=poset,
        rules=[R1, R2, R3],
        bottom_label="non_compliant",
        violation_label="non_compliant",  # demote to bottom on explicit violations
        mutual_exclusions=[mex],
    )

    res = engine.evaluate(kb=kb, entities=[E1])

    print("Final labels:", res.entity_labels)
    print("\nDerivations:")
    for d in res.derivations:
        print(f" - {d.entity}: {d.from_label} -> {d.to_label} by {d.rule_id}")
        for p, a in d.supporting_facts:
            print(f"    support: {p}{a}")
        for p, a in d.mandatory_facts:
            print(f"    mandatory: {p}{a}")
        for n, h, v, desc in d.aggregates:
            print(f"    agg: {n} holds={h} value={v} ({desc})")
    if res.contradictions:
        print("\nContradictions:")
        for e, msg in res.contradictions:
            print(f" - {e}: {msg}")
