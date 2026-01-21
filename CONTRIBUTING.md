# Contributing to IMRA-1

Thank you for pushing frontier reasoning systems forward! This project welcomes research-
quality contributions spanning differentiable programming, meta-cognition, evaluation, and
infrastructure. Please review the guidelines below before opening an issue or pull request.

## Code of Conduct

This project follows the [Code of Conduct](CODE_OF_CONDUCT.md). By participating you agree to
uphold a welcoming, inclusive environment.

## Development Workflow

1. **Fork & Clone** – create a feature branch named `feature/<topic>` or `research/<topic>`.

2. **Bootstrap** – run `make bootstrap` to create a Poetry-managed virtualenv and install deps.
3. **Coding Standards** – adhere to `ruff` + `mypy` static checks; keep modules pure and
   composable; prefer typed dataclasses for structured artifacts.
4. **Testing** – add or update tests under `tests/` for any code change. Large research features
   should also include trace fixtures in `tests/traces/`.
5. **Documentation** – update module guides in `docs/` and add runnable examples when possible.
6. **Pull Request** – reference related issues, describe evaluation impact, and attach reasoning
   traces if relevant.

## Commit Hygiene

- Keep commits focused and descriptive; include design rationale when introducing new modules.
- Reference the metrics or datasets touched, e.g., `reasoning: add differentiable min/max ops`.
- Run `make lint test` before pushing to ensure CI parity.

## Issue Labels

| Label                | Description                                           |
|----------------------|-------------------------------------------------------|
| `type:research`      | Conceptual or experimental contributions              |
| `type:infra`         | Tooling, CI/CD, packaging, or performance work        |
| `module:perceive`    | Intuition engine, sketching, priors                   |
| `module:reason`      | Executor, operators, search loops                    |
| `module:verify`      | Critics, calibration, self-reflection                |
| `module:tracing`     | Trace schemas, persistence, visualization            |
| `good-first-issue`   | Well-scoped starter tasks                            |

## Review Expectations

- Two approvals are required for core module changes (perceive, reason, verify).

- Non-trivial algorithmic changes must include benchmarking notes and failure cases.
- Trace outputs and evaluation notebooks should be attached as artifacts in PR discussions.

Thanks for helping IMRA-1 become a flagship open research framework. This is just one step towards what I am interested in having in the future
