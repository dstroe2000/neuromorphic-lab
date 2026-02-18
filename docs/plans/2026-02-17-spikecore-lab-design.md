# SpikeCore Lab — Design Document

**Date:** 2026-02-17
**Status:** Implemented

## Context

This project bridges the gap between Daniel's SoC/embedded experience and AI/LLM work — specifically **model optimization, graph compilation, and hardware-targeted code generation**. It creates a fully working development lab that compiles a PyTorch model through TVM's BYOC framework to a fictional neuromorphic hardware target ("SpikeCore"), running in a Jupyter notebook where every compilation stage is visible and inspectable.

This serves as onboarding preparation for the Intel AI Software Architect – Neuromorphic Computing role.

## Architecture

### SpikeCore Hardware Model (Fictional)

- 128 neuron cores, each with 256-byte local weight memory
- Integer-only (int8 weights, int16 accumulator)
- 3 primitive ops: `ACC` (weighted accumulate), `FIRE` (threshold + emit spike), `LEAK` (membrane decay)
- Event-driven execution, no global memory bus

### Compilation Pipeline

```
PyTorch Model (float32)
    │
    ▼
TVM Relay IR          ← relay.frontend.from_pytorch()
    │
    ▼
Quantized Relay IR    ← relay.quantize (float32 → int8)
    │
    ▼
Partitioned Graph     ← MergeComposite + AnnotateTarget + PartitionGraph
    │
    ├── Host subgraph (flatten, reshape)
    └── SpikeCore subgraphs (dense+relu, dense+bias)
            │
            ▼
    SpikeCore Assembly    ← BYOC codegen callback
            │
            ▼
    SpikeCore Simulator   ← Event-driven execution on virtual hardware
```

### Dual Execution Paths

1. **TVM path** (inside Docker): Full BYOC pipeline with Relay IR transformations
2. **Standalone path** (no TVM): Manual quantization + direct code generation

Both paths produce the same SpikeCore assembly and use the same simulator.

## Key Design Decisions

1. **TVM BYOC over custom LLVM backend** — BYOC is the actual mechanism Intel would use; avoids LLVM rabbit hole
2. **Docker over conda/venv** — TVM source build has many system deps; container makes it reproducible
3. **PyTorch CPU-only** — no GPU needed, simpler Docker build, focuses on the compilation pipeline
4. **MNIST MLP over anything fancier** — just enough to have real weights and meaningful classification
5. **MLIR deferred to Phase 2** — adding torch-mlir doubles build complexity; TVM BYOC already demonstrates the full pipeline
6. **int8 quantization** — mandatory bridge between float PyTorch world and integer-only SpikeCore
7. **Standalone compilation path** — allows testing and demonstration without TVM build

## Verification Criteria

1. **Docker build**: `docker compose build` completes without errors
2. **Unit tests**: `pytest tests/ -v` — all green
3. **Notebook run**: "Run All Cells" completes without errors
4. **Round-trip accuracy**: SpikeCore classification matches PyTorch top-1 on ≥70% of test samples
5. **Visual inspection**: spike raster shows meaningful temporal activity patterns

## Phase 2 Extensions (Future)

- `02_spikecore_mlir.ipynb` — torch-mlir → custom spikecore MLIR dialect → same simulator
- Convolutional model (event-driven convolutions)
- Multi-chip simulation (spike routing between SpikeCore chips)
- Power/latency estimation model
- Comparison with actual Lava/Loihi workflow
