# IMRA-1 Architecture

IMRA-1 unifies five cognitive subsystems into a metacognitive reasoning loop with coherence analysis.

## Module Responsibilities

| Module | Location | Description |
|--------|----------|-------------|
| PERCEIVE | `imra.core.layers` | Analyzes problem structure, identifies approach, flags uncertainties |
| REASON | `imra.core.layers` | Solves step-by-step with confidence tracking |
| VERIFY | `imra.core.layers` | Checks validity, finds errors, proposes corrections |
| DISCUSS | `imra.core.layers` | Multi-agent debate to challenge assumptions |
| RESOLVE | `imra.core.layers` | Forces concrete answer after discussion |
| GESTALT | `imra.core.gestalt` | Coherence scoring and failure mode detection |
| ORCHESTRATOR | `imra.core.orchestrator` | Coordinates phases and implements feedback loop |
| TRACE | `imra.core.trace` | Captures full audit trail |

## Execution Flow

1. **Problem Input** - Problem is parsed into structured format

2. **PERCEIVE** - Analyzes problem, identifies approach and uncertainties
3. **REASON** - Solves step-by-step, proposes answer with confidence
4. **VERIFY** - Checks solution, finds errors, proposes corrections
5. **DISCUSS** - (Optional) Multiple agents debate the answer
6. **RESOLVE** - (Optional) Forces concrete answer after debate
7. **GESTALT** - Computes coherence score, detects failure modes
8. **FEEDBACK** - Retries with guidance if coherence is low, calibrates confidence
9. **OUTPUT** - Returns final answer with calibrated confidence and trace

## Coherence Scoring

The Gestalt coherence score measures internal consistency:

```python
coherence = (
    0.4 * (1 - discussion_entropy) +
    0.3 * (1 - error_density) +
    0.3 * (1 - confidence_mismatch)
)
```

### Coherence Levels

| Level | Score Range | Meaning |
|-------|-------------|---------|
| BROKEN | < 0.3 | Severe reasoning failures |
| UNCERTAIN | 0.3 - 0.6 | Notable inconsistencies |
| ACCEPTABLE | 0.6 - 0.8 | Minor issues, generally sound |
| COHERENT | > 0.8 | Strong internal consistency |

## Failure Modes

| Mode | Description | Detection |
|------|-------------|-----------|
| VERIFICATION_PARADOX | Claims invalid but no specific errors | High errors, no corrections |
| CIRCULAR_REASONING | Uses conclusion to prove itself | Self-referential patterns |
| INCOMPLETE_REASONING | Gaps in logic chain | Missing steps, no concrete answer |
| META_RETREAT | Discusses methodology instead of solving | Keywords like "methodology", "approach" |
| HEALTHY | Sound reasoning structure | No issues detected |

## Outputs

- **Machine Traces** - JSON files under `research_data/traces/`

- **Coherence Reports** - JSON files with coherence analysis
- **Human-Readable** - Text format traces for debugging

See `src/imra/core/*.py` for implementation details.
