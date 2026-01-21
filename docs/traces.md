# Trace Specification

IMRA-1 captures every reasoning session as a CognitiveTrace with full audit trail.

## Trace Structure

Each trace contains:
- Problem metadata (id, text, expected answer)
- Reasoning steps with timing and confidence
- Errors detected and corrections made
- Final answer and calibrated confidence
- Coherence analysis and failure mode

## Storage Layout

```
research_data/
  traces/
    trace_<id>.json           # Full trace data
    trace_<id>.txt            # Human-readable format
    coherence_<id>.json       # Coherence analysis
```

## Trace Fields

| Field | Type | Description |
|-------|------|-------------|
| trace_id | string | Unique identifier |
| problem | object | Problem metadata |
| reasoning_steps | array | Steps from each phase |
| total_errors_detected | int | Errors found during verification |
| total_corrections_made | int | Errors that were corrected |
| recovery_rate | float | corrections / errors detected |
| final_answer | string | The resolved answer |
| final_confidence | float | Calibrated confidence score |
| coherence_score | float | Gestalt coherence (0-1) |
| coherence_level | string | BROKEN, UNCERTAIN, ACCEPTABLE, COHERENT |
| failure_mode | string | Detected failure pattern |

## Example Trace

```json
{
  "trace_id": "20260119_123456_abc123",
  "problem": {
    "id": "math_001",
    "problem": "Find the sum of divisors of 12",
    "expected": "28"
  },
  "reasoning_steps": [
    {
      "phase": "perceive",
      "confidence": 0.85,
      "uncertainties": ["divisor definition"]
    },
    {
      "phase": "reason",
      "confidence": 0.90,
      "answer": "28"
    },
    {
      "phase": "verify",
      "confidence": 0.75,
      "errors_detected": 0
    }
  ],
  "final_answer": "28",
  "final_confidence": 0.28,
  "original_confidence": 0.50,
  "coherence_score": 0.55,
  "coherence_level": "uncertain",
  "failure_mode": "incomplete_reasoning"
}
```

## Coherence Analysis

Stored separately for detailed analysis:

```json
{
  "coherence_score": 0.55,
  "coherence_level": "uncertain",
  "components": {
    "discussion_entropy": 0.3,
    "error_density": 0.1,
    "confidence_mismatch": 0.2
  },
  "diagnosis": {
    "failure_mode": "incomplete_reasoning",
    "description": "Reasoning has gaps",
    "recommendation": "Provide complete step-by-step solution"
  }
}
```
