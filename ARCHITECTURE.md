# IMRA-1: Metacognitive Reasoning Framework

> **A research system that teaches AI to think about its own thinking.**

IMRA-1 adds a metacognitive layer to AI reasoning. Rather than just producing answers, it analyzes the reasoning process to detect failures, calibrate confidence, and know when it doesn't know.

---

## Architecture


### The Five Layers

| Layer | Purpose |
|-------|---------|
| **PERCEIVE** | Analyzes problem structure, identifies approach, flags uncertainties |
| **REASON** | Solves step-by-step, tracks confidence at each step |
| **VERIFY** | Checks solution validity, finds errors, proposes corrections |
| **DISCUSS** | Multi-agent debate to challenge assumptions (optional) |
| **RESOLVE** | Forces concrete answer after discussion (optional) |

### Gestalt Coherence

The key innovation. Computes a coherence score from:
- Shannon entropy of confidence progression

- Error density (errors found vs steps taken)
- Confidence mismatch (raw confidence vs verification agreement)

```
coherence = 0.4 * (1 - entropy) + 0.3 * (1 - error_density) + 0.3 * (1 - confidence_mismatch)
```

### Failure Modes Detected

| Mode | Description |
|------|-------------|
| **Verification Paradox** | Says answer is wrong but can't find specific errors |
| **Circular Reasoning** | Uses conclusion to prove itself |
| **Incomplete Reasoning** | Gaps in logic, doesn't reach concrete answer |
| **Meta-Retreat** | Explains methodology instead of solving |
| **Healthy** | Reasoning structure is sound |

---

## Key Differentiator

IMRA-1's value lies not in getting answers right, but in **exposing how AI reasons**:

| Traditional LLM | IMRA-1 |
|-----------------|--------|
| Input -> Answer | Input -> PERCEIVE -> REASON -> VERIFY -> Coherence Analysis |
| "The answer is X" | "I tried Y approach with 0.7 confidence, found error, corrected to X" |
| No insight into failures | Full audit trail of every failure and recovery |
| Often confidently wrong | Calibrates confidence: `calibrated = raw * coherence` |

---

## Research Metrics

### 1. Coherence Score
*How internally consistent is the reasoning?*
- Discussion entropy

- Error density
- Confidence mismatch

### 2. Confidence Calibration
*How well does the AI know when it's wrong?*
- Raw confidence vs calibrated confidence

- Overconfidence reduction on wrong answers

### 3. Recovery Rate
*How often does it detect and fix its own mistakes?*
- Errors detected / total errors

- Errors corrected / errors detected

### 4. Failure Mode Distribution
*What kinds of reasoning failures occur?*
- Verification paradox rate

- Circular reasoning rate
- Meta-retreat frequency

---

## Running the Framework

### Quick Demo
```bash
python examples/demo.py
```

### Full Validation Suite
```bash
python run_validation_suite.py
```

### Resume from Specific Problem
```bash
python run_validation_suite.py --start-from 5
```

---

## File Structure

```
IMRA-1/
├── src/imra/
│   ├── core/
│   │   ├── orchestrator.py   # Main reasoning loop + feedback
│   │   ├── layers.py         # PERCEIVE, REASON, VERIFY, DISCUSS, RESOLVE
│   │   ├── gestalt.py        # Coherence analysis + failure detection
│   │   └── trace.py          # Cognitive trace logging
│   └── llm/                  # LLM client abstractions
├── examples/
│   └── demo.py               # Quick demonstration
├── datasets/
│   └── hard_math_problems.json
└── tests/                    # Unit tests
```

---

## Sample Output

```
[PERCEIVE] Analyzing problem...
  [+] Confidence: 0.85
  [+] Uncertainties: 1

[REASON] Solving...
  [+] Proposed answer: 28
  [+] Confidence: 0.90

[VERIFY] Checking solution...
  [+] Valid: True
  [+] Errors found: 0

[GESTALT] Computing coherence score...
  Coherence: 0.55 (UNCERTAIN)
  Failure mode: INCOMPLETE REASONING

RESULT
  Final Answer: 28
  Raw Confidence: 0.50
  Calibrated Confidence: 0.28
```

---

## Research Value

Even if IMRA-1 gets some answers wrong, its value lies in:

1. **Exposing reasoning patterns** - See exactly how AI thinks

2. **Measuring self-awareness** - Quantify confidence calibration
3. **Identifying failure modes** - Detect verification paradox, circular reasoning, etc.
4. **Metacognitive feedback** - Retry with targeted guidance when coherence is low
5. **Full auditability** - Every step logged for inspection

---

## Future Directions (In the works though)

1. **Better base models** - Test with stronger LLMs (GPT-4, Claude, etc.)

2. **Training from traces** - Use failure diagnostics to fine-tune models
3. **Ensemble methods** - Multiple coherence estimators for robustness
4. **Human-in-the-loop** - Surface low-coherence answers for review

---

## License

GNU General Public License v3.0 - See [LICENSE](LICENSE)
