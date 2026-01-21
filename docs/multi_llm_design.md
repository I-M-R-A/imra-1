# IMRA-1 Multi-LLM Design

## Summary

IMRA-1 uses multiple LLM calls across its reasoning layers. Each layer can use a different model, allowing optimization for specific tasks.

## Architecture

```
PERCEIVE (analysis)     -> qwen2.5-coder:3b
REASON (solving)        -> qwen2.5-coder:7b
VERIFY (checking)       -> codellama:latest
DISCUSS (debate)        -> qwen2.5-coder:7b
RESOLVE (synthesis)     -> qwen2.5-coder:7b
```

## Components

- `imra.llm.OllamaClient` - HTTP wrapper for Ollama endpoints

- `imra.core.orchestrator.ReasoningOrchestrator` - Coordinates phases
- `imra.core.layers.*Layer` - Individual layer implementations

## Configuration

Models are configured in `configs/` or via environment:

```bash
export OLLAMA_URL=http://localhost:11434
```

Default models are set in `src/imra/llm/client.py`.

## Running

```bash
python examples/demo.py # Quick Demo

python run_validation_suite.py # Running all full problems
```

## Supported Backends

Currently supported:
- Ollama (local LLM inference)

Planned in the future for more accuracy:
- OpenAI API

- Anthropic API
- vLLM
- DeepSeek V3.2

## Model Selection

Recommendations by layer:
- **PERCEIVE**: Smaller, faster model (problem analysis is simpler)

- **REASON**: Larger model (complex reasoning benefits from capability)
- **VERIFY**: Capable model (error detection requires understanding)
- **DISCUSS/RESOLVE**: Same as REASON
