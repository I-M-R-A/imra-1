# Verify Module

The VERIFY layer checks solutions and identifies errors.

## Responsibilities

1. Check the proposed solution from REASON
2. Identify specific errors in reasoning
3. Propose corrections when errors are found
4. Enforce that errors must be acknowledged

## Key Components

- `VerifyLayer` - Main class in `imra.core.layers`
- `LayerOutput` - Structured output with validity, errors, and corrections

## Output Fields

| Field | Type | Description |
|-------|------|-------------|
| content | string | Verification analysis |
| confidence | float | Confidence in verification (0-1) |
| errors | list | Detected errors with descriptions |
| raw_response | dict | Full response including is_valid and corrected_answer |

## Example Output

```
[VERIFY] Checking solution...
  [+] Valid: True
  [+] Errors found: 0
```

Or with errors:

```
[VERIFY] Checking solution...
  [+] Valid: False
  [+] Errors found: 2
  [+] Corrected to: 42
```

## Error Enforcement

The VERIFY layer enforces that identified errors must be acknowledged. If errors are found but no corrections proposed, this indicates a verification paradox (detected by GESTALT).

## Implementation

Located in `src/imra/core/layers.py`. Uses LLM with structured prompts to check solutions and identify specific errors.
