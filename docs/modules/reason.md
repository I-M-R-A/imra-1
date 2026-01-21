# Reason Module

The REASON layer solves problems step-by-step.

## Responsibilities

1. Take the analysis from PERCEIVE
2. Execute the solution approach step-by-step
3. Track confidence at each step
4. Produce a proposed answer

## Key Components

- `ReasonLayer` - Main class in `imra.core.layers`
- `LayerOutput` - Structured output with answer and confidence

## Output Fields

| Field | Type | Description |
|-------|------|-------------|
| content | string | Step-by-step reasoning |
| confidence | float | Confidence in the answer (0-1) |
| raw_response | dict | Full LLM response including answer field |

## Example Output

```
[REASON] Solving...
  [+] Proposed answer: 28
  [+] Confidence: 0.90
```

## Implementation

Located in `src/imra/core/layers.py`. Uses LLM to solve problems with structured prompts that encourage step-by-step reasoning.
