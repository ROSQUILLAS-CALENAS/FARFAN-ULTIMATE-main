#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Appends an innovative "Advanced Graphics Stack" section with Mermaid diagrams to all README-like files,
if not already present.

Idempotency: guarded by marker <!-- ADV_GRAPHICS_STACK:BEGIN v1 --> ... <!-- ADV_GRAPHICS_STACK:END v1 -->
"""
import os
# # # from pathlib import Path  # Module not found  # Module not found  # Module not found

ROOT = Path(__file__).resolve().parents[1]
MARKER_BEGIN = "<!-- ADV_GRAPHICS_STACK:BEGIN v1 -->"
MARKER_END = "<!-- ADV_GRAPHICS_STACK:END v1 -->"

BLOCK = (
    MARKER_BEGIN
    + """
## Advanced Graphics Stack — Innovative Holistic Visuals

This section provides a multi-perspective, advanced visualization of the EGW Query Expansion and Hybrid Retrieval system. The diagrams are designed to be composable, auditable, and implementation-agnostic.

### 1) System Holomap (Architecture Overview)
```mermaid
flowchart LR
  %% Clusters
  subgraph Retrieval["Hybrid Retrieval"]
    A[User Query] --> B{Router ρ}
    B -->|Sparse| S[(SPLADE/BM25)]
    B -->|Dense| D[(E5/FAISS)]
    B -->|Late| L[(ColBERTv2)]
  end
  subgraph OT["EGW Optimal Transport"]
    Q[Query Graph] --- C[Corpus Graph]
    Q -->|ε, λ| T[Transport Plan Π*]
  end
  subgraph Assurance["Contracts & Risk"]
    V1[[Routing Contract ρ]]
    V2[[Conformal Risk]]
    V3[[MCC/BMC]]
  end

  A --> Q
  S & D & L --> C
  T --> R[Expanded Canonical Queries]
  R --> Index[Hybrid Index]
  R --> Answer[Answer Synthesizer]
  V1 -. audits .- B
  V2 -. certify .- Answer
  V3 -. certify .- Pipeline
```

### 2) Deterministic Routing Sequence
```mermaid
sequenceDiagram
  participant U as User
  participant R as Deterministic Router ρ
  participant OT as EGW Engine
  participant H as Hybrid Index
  participant V as Validators (ρ, MCC/BMC, Conformal)
  participant S as Synthesizer
  U->>R: q, Θ, σ, κ
  R->>OT: pattern alignment(q)
  OT-->>R: Π*, barycenter
  R->>H: retrieve(sparse | dense | late)
  H-->>R: candidates
  R->>V: attest(trace, hashes)
  V-->>R: certificate
  R->>S: context, certificate
  S-->>U: answer + lineage
```

### 3) Evidence/Context State Machine
```mermaid
stateDiagram-v2
  [*] --> Ingested
  Ingested --> Normalized
  Normalized --> Indexed: hybrid
  Indexed --> Routed: A* deterministic
  Routed --> Expanded: OT barycenter
  Expanded --> Validated: contracts + conformal
  Validated --> Answered
  Answered --> [*]
```

### 4) Component Metamodel
```mermaid
classDiagram
  class Router {
    +route(q, Θ, σ, κ)
    +deterministicAStar()
    +trace()
  }
  class EGWEngine {
    +align(Q, C)
    +barycenter()
    +stability(ε, λ)
  }
  class HybridIndex {
    +sparse()
    +dense()
    +lateInteraction()
  }
  class Validator {
    +routingContract()
    +conformalRisk()
    +mcc_bmc()
  }
  class Synthesizer {
    +compose()
    +verifyLineage()
  }
  Router --> EGWEngine
  Router --> HybridIndex
  Router --> Validator
  Router --> Synthesizer
```

### 5) Retrieval ER Model
```mermaid
erDiagram
  QUERY ||--o{ CANONICAL_QUERY : expands_to
  CANONICAL_QUERY }o--|| INDEX : indexed_in
  QUERY }o--o{ EVIDENCE : supports
  EVIDENCE ||--o{ CERTIFICATE : yields
  CERTIFICATE ||--|| CONTRACT : attests
  INDEX ||--o{ RETRIEVAL : produces
  RETRIEVAL }o--|| SYNTHESIS : feeds
```

### 6) Pipeline Timeline (Gantt)
```mermaid
gantt
  dateFormat  X
  title Pipeline Phases (Logical Timeline)
  section Preparation
  Ingest/Normalize        :a1, 0, 10
  Build Graphs (Q,C)      :a2, 10, 30
  section Alignment
  EGW Transport Solve     :b1, 30, 50
  Barycenter              :b2, 50, 60
  section Retrieval
  Sparse/Dense/Late       :c1, 60, 80
  section Assurance
  Contracts + Conformal   :d1, 80, 90
  section Synthesis
  Answer + Lineage        :e1, 90, 100
```

> Tip: GitHub renders Mermaid in Markdown by default. If a specific viewer does not support Mermaid, consider using Mermaid live editors to export PNG/SVG equivalents.

"""
    + MARKER_END
)


def is_readme_like(path: Path) -> bool:
    name = path.name
    lname = name.lower()
    if path.is_dir():
        return False
    # Common README patterns
    patterns = [
        "readme",  # README, readme
    ]
    return any(p in lname for p in patterns)


def collect_targets(root: Path):
    targets = []
    for dirpath, dirnames, filenames in os.walk(root):
        # skip virtualenvs and .git
        if any(skip in dirpath for skip in (os.sep+".venv", os.sep+".git", os.sep+"__pycache__")):
            continue
        for fn in filenames:
            p = Path(dirpath) / fn
            if is_readme_like(p):
                targets.append(p)
    return targets


def apply_block(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # skip non-utf8
        return False
    if MARKER_BEGIN in text:
        return False
    # Ensure file ends with a newline before appending
    sep = "\n" if not text.endswith("\n") else ""
    new_text = text + sep + "\n" + BLOCK + "\n"
    path.write_text(new_text, encoding="utf-8")
    return True


def main():
    targets = collect_targets(ROOT)
    changed = []
    for p in targets:
        if apply_block(p):
            changed.append(str(p.relative_to(ROOT)))
    print(f"Updated {len(changed)} file(s).")
    if changed:
        print("\n".join(changed))


if __name__ == "__main__":
    main()
