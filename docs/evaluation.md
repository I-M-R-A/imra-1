# Evaluation Protocol

IMRA-1 emphasizes reasoning quality and self-awareness over raw accuracy.

## Core Metrics

### 1. Coherence Score
Measures internal consistency of reasoning:
```
coherence = 0.4 * (1 - entropy) + 0.3 * (1 - error_density) + 0.3 * (1 - confidence_mismatch)
```

### 2. Confidence Calibration
How well the system knows when it's wrong:
- Raw confidence vs calibrated confidence

- Overconfidence reduction on incorrect answers
- Calibration formula: `calibrated = raw * coherence`

### 3. Recovery Rate
Self-correction capability:
- Errors detected / total errors

- Errors corrected / errors detected

### 4. Failure Mode Distribution
Types of reasoning failures:
- Verification paradox rate

- Circular reasoning rate
- Incomplete reasoning rate
- Meta-retreat frequency

## Test Battery

### 1. Core Functional Coverage
- **PERCEIVE**: Problem analysis accuracy, uncertainty detection

- **REASON**: Step-by-step solution quality, confidence tracking
- **VERIFY**: Error detection precision/recall, correction accuracy

### 2. Coherence Analysis
- Gestalt scoring accuracy

- Failure mode detection precision
- Feedback loop effectiveness

### 3. Challenging Problems
- Math olympiad problems

- Logic puzzles
- Problems designed to trigger specific failure modes

## Running Evaluation

```bash
python run_validation_suite.py

pytest tests/unit/

pytest tests/unit/core/test_gestalt.py -v
```

## Output Metrics

Each run produces:
- Accuracy (correct / total)

- Average coherence score
- Failure mode distribution
- Confidence calibration stats
- Recovery rate

## Interpreting Results

| Metric | Good | Needs Work |
|--------|------|------------|
| Coherence | > 0.6 | < 0.4 |
| Recovery Rate | > 0.5 | < 0.2 |
| Calibration Gap | < 0.2 | > 0.4 |

Note: Low accuracy with good coherence indicates the base LLM needs improvement, not the framework.
