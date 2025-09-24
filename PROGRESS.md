# Project Progress Log

This document tracks implemented functions, their responsibilities, and current progress.

## Functions

- **app/core/orchestrator.py::handle_user**
  - Description: Entry point for the main orchestration pipeline handling user messages.
  - Progress: Stub created with updated TODO outlining the eight pipeline steps.

- **app/core/neuro.py**
  - Description: Neurotransmitter state management utilities (snapshot, set_levels, bias_to_style, sleep_reset).
  - Progress: Module scaffold with docstring and stubs awaiting implementation.

- **app/core/bandit.py**
  - Description: Global ε-greedy bandit for selecting suggestions across intents and kinds.
  - Progress: Stub functions for pick/update prepared for future logic.

- **app/core/router.py**
  - Description: RAG routing with primary search and e5 reranking helpers.
  - Progress: Stubs for search and rerank_e5 awaiting implementation.

- **app/core/budget.py::trim**
  - Description: Token budget allocator across history, RAG hits, and junior metadata.
  - Progress: Stub ready for prioritization logic.

- **app/core/guard.py::soft_censor**
  - Description: Soft censorship pass before delivering responses (profanity/PII masking).
  - Progress: Stub placeholder for future filtering implementation.

- **app/core/tools_runner.py::run_all**
  - Description: Executes senior-issued tool calls via dynamic loading.
  - Progress: Stub function marked for later development.

- **app/core/env_state.py**
  - Description: Environment session tracking and brief construction helpers.
  - Progress: Stubs established for ensuring sessions and building briefs.

- **app/agents/junior.py::generate**
  - Description: Generates junior JSON v2 dispatcher output based on conversation context.
  - Progress: Stub awaiting LLM integration.

- **app/agents/senior.py::generate_structured**
  - Description: Produces structured senior replies including text, tool calls, memory, and plans.
  - Progress: Stub prepared for model wiring.

- **app/tools/**
  - Description: Tool entrypoints for notes, reminders, aliases, Telegram messaging, and date search.
  - Progress: Each tool exposes a stubbed function raising NotImplementedError.

- **scripts/consolidate_sleep.py::run_sleep_batch**
  - Description: Batch consolidator for sleep cycles, RAG updates, and model refresh operations.
  - Progress: Stub prepared per specification.

## Notes

- Core configuration files (`config/llm.yaml`, `config/persona.yaml`, `config/router.yaml`, `config/tools.yaml`) and JSON schemas were updated to match the new Arkestra skeleton specification.
- Directory scaffolding, package initializers, and placeholder modules were created to align with the initialization prompt.
