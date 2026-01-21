# Perceive Module

The PERCEIVE layer analyzes problems and identifies the approach.

## Responsibilities

1. Parse and understand the problem statement

2. Identify the type of problem and appropriate approach
3. Flag potential uncertainties and ambiguities
4. Produce structured output for the REASON layer

## Key Components

- `PerceiveLayer` - Main class in `imra.core.layers`

- `LayerOutput` - Structured output with content, confidence, errors, alternatives

## Output Fields

| Field | Type | Description |
|-------|------|-------------|
| content | string | Analysis of the problem |
| confidence | float | Confidence in the analysis (0-1) |
| alternatives | list | Alternative approaches considered |
| uncertainties | list | Identified ambiguities or gaps |

## Example Output

```
[PERCEIVE] Analyzing problem...
  [+] Confidence: 0.85
  [+] Alternatives: 2
  [+] Uncertainties: 1
```

## Implementation

Located in `src/imra/core/layers.py`. Uses LLM to analyze problem and extract structured information.
