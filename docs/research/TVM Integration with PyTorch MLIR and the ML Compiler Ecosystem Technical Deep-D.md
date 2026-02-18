# TVM — Integration with PyTorch, MLIR, and the ML Compiler Ecosystem: Technical Deep-Dive

## Key Questions This Report Answers
1. What is TVM's architecture and how does its multi-level IR stack (Relay, Relax, TIR) enable hardware-portable optimization?
2. How does TVM integrate with PyTorch — through torch.compile, the Relax frontend, or both?
3. What is the relationship between TVM and MLIR, and are they competitors or complementary infrastructure?
4. How does torch-mlir bridge PyTorch to the MLIR compiler ecosystem, and how does this compare to TVM's PyTorch path?
5. Where does TVM fit in the broader ML compiler landscape alongside XLA, IREE, TorchInductor, and OpenXLA?
6. What are TVM's production deployment patterns, and where does it outperform or underperform alternatives?
7. What does TVM's future look like after NVIDIA's acquisition of OctoML and the rise of torch.compile?

## The Mental Model (How to Think About This)

Think of an ML compiler as a translator chain: your PyTorch model speaks "Python," but your hardware — whether it is an NVIDIA GPU, an Apple M-series chip, a Qualcomm mobile SoC, or an FPGA — speaks its own low-level dialect. The compiler's job is to translate from one to the other, and to find clever rephrasing along the way that makes the message faster and more efficient.

TVM is like a full-service translation agency. You hand it a model in PyTorch (or TensorFlow, or ONNX), and it passes the work through two internal desks: the **graph desk** (Relay or Relax) reorganizes the overall structure — fusing operations, eliminating redundancy, picking the best data layout — and the **loop desk** (TIR) rewrites each individual operation into tight, hardware-specific code with optimal memory access patterns, parallelism, and vectorization. Crucially, TVM employs a machine-learning-based search (auto-tuning) to explore thousands of possible translations and pick the fastest one for your specific hardware.

MLIR, by contrast, is not a translation agency but a modular workbench for building translation tools. It provides standardized building blocks — called "dialects" — that compiler engineers can assemble into custom pipelines. The OpenXLA consortium (led by Google, with Intel, Apple, NVIDIA, and others) uses MLIR to build XLA, IREE, and the StableHLO exchange format. Meanwhile, PyTorch's own torch.compile pipeline (with TorchInductor as the default backend) represents a third path that has become dominant for NVIDIA GPU users.

The ML compiler world is thus a three-body system: TVM, the MLIR/OpenXLA ecosystem, and torch.compile/TorchInductor — each with different strengths, and PyTorch sitting at the center as the framework everyone wants to compile. TVM's competitive advantage is clearest at the hardware periphery: mobile, edge, WebGPU, and custom accelerators, where TorchInductor's Triton-centric approach does not reach.

## Prerequisites / What You Need to Know First

This report assumes familiarity with deep learning frameworks (PyTorch in particular), the concept of computational graphs, and basic compiler terminology (intermediate representation, optimization passes, code generation). You should understand that neural network inference involves executing a sequence of tensor operations (matrix multiplications, convolutions, activations) and that different hardware executes these operations through different instruction sets and memory hierarchies.

No prior knowledge of TVM, MLIR, or compiler design is required — all key concepts are introduced progressively. Familiarity with the following will accelerate your reading: Python decorators and metaprogramming (for TVMScript), CUDA programming concepts (for understanding TIR's threading model), and the idea of operator fusion (combining multiple neural network layers into a single GPU kernel to reduce memory traffic).

## TL;DR

Apache TVM is an end-to-end ML compiler that translates framework models into optimized code for diverse hardware through a multi-level IR stack (Relax/Relay for graphs, TIR for loops) and learning-based auto-tuning. It integrates with PyTorch via both the torch.compile backend API and a direct Relax frontend, though TorchInductor dominates for NVIDIA GPUs. TVM's strongest value proposition is cross-platform deployment (mobile, edge, WebGPU) via MLC-LLM, while the MLIR/OpenXLA ecosystem (with torch-mlir, StableHLO, IREE) represents the primary competing infrastructure. NVIDIA's 2024 acquisition of OctoML — TVM's commercial arm — has reshaped the competitive landscape, but TVM remains Apache-governed and technically differentiated for non-NVIDIA targets.

## 1. What Problem Does This Solve?

Modern deep learning faces a hardware fragmentation problem. Models are typically developed in PyTorch on NVIDIA GPUs, but deployment targets span a vast range: cloud GPUs from multiple vendors, mobile processors (ARM, Qualcomm), edge accelerators, FPGAs, and custom ASICs. Each hardware target has its own instruction set, memory hierarchy, and parallelism model. Writing hand-optimized kernels for every operator on every target is prohibitively expensive — and yet naive compilation leaves enormous performance on the table.

The ML compiler's mission is to close this gap automatically. Given a model from a high-level framework, the compiler must: (1) capture the computation as a structured graph, (2) apply graph-level optimizations (operator fusion, constant folding, layout transformation), (3) generate hardware-specific loop-level code for each operator, and (4) auto-tune the generated code to find the fastest configuration for the specific target [1].

TVM, introduced at OSDI '18, was among the first systems to address all four stages in a unified framework [1]. Its key innovation was combining graph-level and operator-level optimization with a **learning-based cost model** that predicts performance without exhaustive hardware measurement, enabling automated search across the vast space of possible code transformations [1,3]. This approach — using ML to optimize ML — remains central to TVM's design philosophy and is formalized in the TVM Unity vision [8].

But TVM does not exist in a vacuum. The problem of ML compilation has attracted enormous investment: Google built XLA for TensorFlow and JAX, then open-sourced it as OpenXLA with IREE and StableHLO [12]; LLVM created MLIR as general-purpose compiler infrastructure [4,5]; PyTorch developed torch.compile with TorchInductor as its default backend [9]; and NVIDIA, Intel, Qualcomm, and others have built proprietary optimization stacks. Understanding where TVM fits — and where it excels — requires understanding this entire ecosystem.

## 2. How It Works

### 2.1 The IRModule: TVM's Core Data Structure

Everything in TVM revolves around the **IRModule**, a container that holds both high-level graph functions (in Relay or Relax) and low-level tensor programs (in TIR) [6]. Compilation in TVM is a sequence of transformations on IRModules: each pass reads an IRModule and produces a new, optimized IRModule. This uniform representation is what enables TVM's cross-level optimization — graph-level and loop-level transformations can inspect and modify the same compilation unit.

Think of the IRModule as a project folder containing two types of documents: architectural blueprints (the graph IR, showing how operators connect) and detailed engineering specs (the TIR programs, showing how each operator executes on hardware). A renovation pass might redraw the blueprints (fuse two rooms into one), rewrite the specs (change the loop order in a matrix multiplication), or do both at once.

### 2.2 High-Level IR: From Relay to Relax

**Relay** is TVM's established high-level functional IR. It represents the computation as a dataflow graph with support for control flow, recursion, and complex data structures [6,1]. Relay enables graph-level optimizations including:

- **Operator fusion**: Combining multiple operators into a single kernel to reduce memory traffic
- **Constant folding**: Pre-computing operations on known values at compile time
- **Dead code elimination**: Removing unused computations
- **Layout transformation**: Choosing data layouts (NCHW vs. NHWC) optimal for the target hardware

**Relax** ("Relay Next") is TVM's next-generation graph IR, introduced in TVM v0.13 and now central to active development (v0.22 RC) [2,7]. Relax addresses a fundamental limitation of Relay: static shape assumptions. In LLM inference, sequence lengths vary dynamically — a chat model processes different-length prompts. Relay required shape specialization, meaning separate compilation for each possible shape. Relax introduces **first-class symbolic shape annotations** that track shapes algebraically (e.g., `batch_size * seq_len`) without sacrificing optimization opportunities [2].

Relax also implements the **TVM Unity** vision of cross-level abstraction [8]. A single Relax function can encapsulate computational graph nodes, loop-level TIR programs, and calls to external libraries (cuBLAS, cuDNN) in a unified representation [2]. This enables a qualitatively new capability: co-optimization across abstraction levels. For example, the compiler can decide whether to fuse two operators into a single TIR kernel or to call separate library functions, based on end-to-end performance analysis.

Benchmarks show Relax achieving up to 27% reduction in decode token latency compared to alternatives, with performance close to TVM's static graph runtime even with dynamic shapes [2]. *Note: this claim derives from a single source (the Relax paper) and has not been independently replicated across diverse model families.*

### 2.3 Low-Level IR: TIR and Code Generation

**TIR (Tensor Intermediate Representation)** is where hardware-specific optimization happens. TIR operates at the loop-nest level with multi-dimensional load/store operations, threading primitives, and vector/tensor instructions [6]. If Relax decides *what* operations to perform and in what order, TIR decides *how* each operation executes on the metal.

TIR transformations come in two forms:
- **TensorIR schedules**: User-guided or auto-tuning-discovered optimization primitives (loop tiling, unrolling, vectorization, thread binding)
- **Lowering passes**: Automated translation toward hardware-specific code

Final code generation targets LLVM IRBuilder for x86/ARM CPUs or produces source-level CUDA C for NVIDIA GPUs and OpenCL for other accelerators [6].

### 2.4 Auto-Tuning: Three Generations of Automated Optimization

Auto-tuning is arguably TVM's most distinctive feature. Rather than relying on fixed heuristics, TVM uses machine learning to search for optimal code transformations. The system has evolved through three generations [1,3,7]:

1. **AutoTVM (1st generation)**: Requires hand-written search space templates for each operator. Experts define what knobs to turn; the system finds the best settings [1].

2. **Ansor/AutoScheduler (2nd generation)**: Eliminates the need for templates entirely. Uses derivation-based sketch generation to automatically construct the search space, then explores it with a learned cost model [3]. This was a major democratization — non-experts could now get competitive performance.

3. **MetaSchedule (3rd generation)**: Unifies the best of both approaches into a single framework with improved search efficiency. Supports both template-guided and template-free search within the same system [7].

All three generations share a core innovation: **learning-based cost models** that predict the execution time of a code variant without actually running it on hardware [1,3]. This makes search tractable — the system can evaluate millions of candidates in the time it would take to benchmark a few hundred.

**DLight** complements MetaSchedule by providing pre-defined, lightweight TIR schedules specifically optimized for LLM GPU workloads with dynamic shape support [6]. DLight enables fast LLM optimization without expensive auto-tuning search, which is critical for MLC-LLM's deployment pipeline.

### 2.5 Supporting Infrastructure

- **TOPI (Tensor Operator Inventory)**: Library of operator templates providing baseline implementations for standard neural network operations [6]
- **BYOC (Bring Your Own Codegen)**: Mechanism for hardware vendors to plug in custom backend code generators without modifying TVM's core [6]
- **TVMScript**: Python-based DSL for writing and inspecting TVM programs, providing a readable syntax for TIR and Relax functions [6]

## 3. Under the Hood: PyTorch Integration Pathways

The relationship between TVM and PyTorch is multifaceted. There are two primary integration paths, each with different trade-offs.

### 3.1 Path 1: TVM as a torch.compile Backend

PyTorch 2.x introduced `torch.compile`, a compilation pipeline with three stages [9]:

1. **TorchDynamo**: A graph capture frontend that uses frame evaluation hooks to extract computation graphs from Python code — even code with control flow, data-dependent shapes, and Python side effects.
2. **AOTAutograd**: Traces the backward pass ahead of time, enabling compilation of both forward and backward computations.
3. **Backend**: Takes the captured graph and generates optimized code. The default backend is **TorchInductor**, which uses OpenAI Triton for NVIDIA/AMD GPUs and C++/OpenMP for CPUs [9].

TVM can serve as an alternative backend in this pipeline. Instead of TorchInductor generating code, TVM's compilation stack (Relay/Relax → TIR → auto-tuned code) processes the captured graph. This integration means users can switch from TorchInductor to TVM with a single line of code:

```python
# Default: TorchInductor
compiled_model = torch.compile(model)

# Alternative: TVM backend
compiled_model = torch.compile(model, backend="tvm")
```

However, there are important caveats. PyTorch's own documentation notes that "other backends are unlikely to beat TorchInductor when running on NVIDIA GPUs" [9]. TVM's torch.compile backend also has known operator coverage gaps, though TVM v0.22 RC includes improvements to the FX-based PyTorch frontend, including bug fixes for in-place operations and PyTorch 2.7 compatibility [7].

### 3.2 Path 2: Relax PyTorch Frontend (Direct Import)

The second path bypasses torch.compile entirely. Relax provides a direct import mechanism from PyTorch via FX graph tracing [2,7]:

```python
import tvm
from tvm.relax.frontend.torch import from_fx

# Trace and import into Relax
fx_module = torch.fx.symbolic_trace(model)
relax_mod = from_fx(fx_module, input_info)
```

This path is used by **MLC-LLM** and other TVM-native deployments [13]. It gives TVM full control over the compilation pipeline — from graph capture through code generation — without depending on PyTorch's compilation infrastructure. The trade-off is that this path does not benefit from TorchDynamo's sophisticated graph capture or AOTAutograd's training support.

### 3.3 When to Use Which Path

The choice between paths depends on the deployment target:

- **NVIDIA GPUs (training or inference)**: torch.compile + TorchInductor is the pragmatic default. TVM is unlikely to beat it [9].
- **Non-NVIDIA hardware**: TVM via either path, as TorchInductor's Triton backend targets NVIDIA/AMD only.
- **Cross-platform deployment (mobile, edge, WebGPU)**: Relax frontend → TVM compilation → platform-specific deployment via MLC-LLM [13].
- **Custom accelerators**: TVM via BYOC, regardless of integration path [6].

## 4. Performance & Benchmarks

### 4.1 TVM/Relax Performance Characteristics

Relax demonstrates competitive performance across deployment scenarios:
- Up to **27% reduction in decode token latency** compared to alternatives, with performance approaching TVM's static graph runtime even with fully dynamic shapes [2] *(single-source claim from the Relax paper — treat as indicative pending broader replication)*
- MLC-LLM (built on Relax + TIR) achieves **best concurrency/batching performance on Apple Silicon** and robust long-context scaling from 32K to 128K tokens [13]

### 4.2 TorchInductor Dominance on NVIDIA

For NVIDIA GPU workloads, torch.compile with TorchInductor has become the reference point:
- Static KV-cache combined with torch.compile yields up to **4x speedup** for LLM inference on HuggingFace models [14] *(single-source benchmark from HuggingFace documentation)*
- TorchInductor benefits from deep Triton integration and NVIDIA's continuous optimization of the CUDA stack

### 4.3 The Benchmark Gap

One significant limitation in the current landscape is the **lack of comprehensive head-to-head benchmarks** between TVM/Relax and TorchInductor across diverse workloads and hardware targets. Most available comparisons focus on specific scenarios (LLM inference on specific GPUs) rather than systematic cross-platform evaluation. This makes definitive performance claims difficult outside of known sweet spots.

### 4.4 AWS SageMaker Neo

AWS SageMaker Neo uses Apache TVM and partner compilers to optimize inference, reporting up to **25x faster inference** across NVIDIA, Intel, ARM, Qualcomm, and Xilinx targets [18]. This provides evidence of TVM's value in heterogeneous deployment environments, though this claim is vendor-reported and independent verification is limited. *(Single-source claim.)*

## 5. Practical Considerations

### 5.1 TVM vs. the MLIR/OpenXLA Ecosystem

MLIR and TVM operate at different levels of abstraction, but their ecosystems compete for the same users [4,10]:

| Dimension | TVM Ecosystem | MLIR/OpenXLA Ecosystem |
|-----------|--------------|------------------------|
| **Core IR** | Relay/Relax + TIR | MLIR dialects + StableHLO |
| **Auto-tuning** | MetaSchedule/Ansor | XLA auto-clustering, IREE dispatch |
| **Hardware breadth** | Broad (CPU, GPU, mobile, FPGA) | Primarily Google TPU + GPU, growing |
| **PyTorch path** | FX import, torch.compile backend | torch-mlir → StableHLO → IREE/XLA |
| **Governance** | Apache Foundation | LLVM + OpenXLA consortium |
| **Commercial backing** | OctoML → NVIDIA (acquired 2024) | Google-led consortium |

The fundamental distinction: TVM is an **end-to-end compiler** (framework → optimized code); MLIR is **compiler infrastructure** (building blocks for creating compilers) [4,10]. Chris Lattner (Modular/LLVM) has argued that "domain-specific approaches (like TVM) don't integrate well with existing compiler infrastructures" [10], while TVM Unity proponents counter that cross-level optimization is more valuable than infrastructure uniformity.

### 5.2 torch-mlir: The Other PyTorch Bridge

**torch-mlir** is an LLVM Incubator project that provides an alternative bridge from PyTorch to the MLIR compiler world [11]. Its architecture converts PyTorch models to the Torch MLIR dialect, with backend lowerings to three targets:

1. **Linalg-on-Tensors**: Most complete path, full dynamic shape support
2. **TOSA (Tensor Operator Set Architecture)**: Hardware-agnostic tensor operations
3. **StableHLO**: For consumption by XLA, IREE, and other OpenXLA components [11,17]

torch-mlir remains an LLVM Incubator project (not yet an official LLVM component) with an active community including weekly meetings and LLVM Discord presence [11,25]. Its significance lies in enabling the pipeline: PyTorch → torch-mlir → StableHLO → IREE/XLA — a completely MLIR-native path that competes with TVM's PyTorch frontend.

### 5.3 The OpenXLA Consortium

OpenXLA represents the most organized effort to standardize ML compilation infrastructure. Co-developed by Alibaba, AWS, AMD, Apple, Arm, Cerebras, Google, Graphcore, HuggingFace, Intel, Meta, NVIDIA, and SiFive [12], it includes:

- **XLA**: The compiler (strongest on TPUs and JAX workloads)
- **StableHLO**: Backward-compatible portability layer [17]
- **IREE**: Execution environment for deployment, including mobile/edge [16]
- **Shardy**: Tensor partitioning for distributed workloads
- **PJRT**: Hardware-independent plugin interface

StableHLO is emerging as a potential **universal exchange format** — it can be produced by TVM, JAX, TensorFlow, and PyTorch (via torch-mlir), and consumed by XLA, IREE, and other compilers [17]. This creates a possible convergence path where TVM and the MLIR ecosystem interoperate through StableHLO rather than compete.

### 5.4 The HuggingFace Gap

A notable practical concern: TVM is largely absent from HuggingFace's optimization pipeline [14]. HuggingFace Optimum supports ONNX Runtime, OpenVINO, ExecuTorch, AWS Neuron, and TensorRT-LLM — but no official `optimum-tvm` package exists. MLC-LLM uses its own model distribution pipeline rather than HuggingFace Hub [13]. For teams whose workflow centers on HuggingFace, this represents a real adoption barrier.

### 5.5 Deployment Ecosystem Positioning

TVM's deployment niche is clearest at the edges of the hardware landscape:

- **Where TVM wins**: Cross-platform deployment (iOS, Android, WebGPU, embedded), non-NVIDIA targets, custom accelerators via BYOC, dynamic shape workloads via Relax [2,6,13]
- **Where alternatives win**: NVIDIA-only inference (TorchInductor, TensorRT-LLM), Google TPU workloads (XLA/JAX), HuggingFace-centric workflows (ONNX Runtime, Optimum), Intel hardware (OpenVINO) [9,14,21]

### 5.6 OctoML Acquisition: Impact on TVM's Future

OctoML — the commercial company founded by TVM's original creators to provide TVM-based optimization-as-a-service — was acquired by NVIDIA in 2024 [15]. The acquisition is significant for several reasons: it removes the primary commercial entity funding TVM-specific engineering, raises questions about NVIDIA's motivation to promote a compiler that benefits non-NVIDIA hardware, and may accelerate the shift of TVM's performance lead toward NVIDIA targets. However, TVM remains Apache-governed, and MLC-LLM (the most visible TVM consumer) is independently funded. The long-term trajectory depends on whether Apache community contributions can replace OctoML's engineering investment.

## 6. Implementation Roadmap

### Phase 1: Evaluation and Prototyping
- Install TVM from source or pre-built packages; explore TVMScript with tutorial models
- Benchmark a representative model on your target hardware using both TVM (Relax frontend) and TorchInductor
- Assess operator coverage for your model architecture — check for unsupported ops that would fall back to unoptimized execution
- Evaluate MLC-LLM if your use case involves LLM deployment on non-NVIDIA targets

### Phase 2: Integration Development
- Choose your PyTorch integration path: torch.compile backend (easier migration, shared infrastructure) vs. Relax frontend (full TVM control, better for non-NVIDIA deployment)
- Implement auto-tuning with MetaSchedule for your target hardware; budget 2–8 hours of tuning time per model-hardware combination
- Set up CI/CD pipelines that include TVM compilation as a build step
- Explore BYOC if you have custom hardware targets

### Phase 3: Production Deployment
- Deploy using MLC-LLM for cross-platform scenarios or TVM's C++ runtime for embedded targets
- Implement performance monitoring to compare TVM-compiled models against baselines
- Establish a re-tuning cadence as models and hardware evolve
- Consider contributing upstream: TVM is Apache-governed and benefits from community operator coverage expansion [7]

## 7. Getting Started

The fastest path to a working TVM deployment depends on your scenario:

**For LLM deployment on diverse hardware**: Start with MLC-LLM [13]. It provides pre-built packages, model conversion scripts, and deployment runtimes for iOS, Android, Web (WebGPU), and desktop. This is TVM's most polished user experience.

**For custom model compilation**: Install TVM from source following the official documentation at tvm.apache.org [6]. Use the Relax frontend to import your PyTorch model, apply graph optimizations, and run MetaSchedule auto-tuning for your target. The TVMScript tutorials provide worked examples of the full pipeline.

**For torch.compile integration**: If you are already using torch.compile, simply install the TVM backend package and pass `backend="tvm"` to `torch.compile()`. This requires minimal code changes but may have operator coverage limitations [7,9].

**For evaluating the landscape**: Before committing to TVM, benchmark your specific workload against TorchInductor (for NVIDIA GPUs), ONNX Runtime (for broad compatibility), and OpenVINO (for Intel hardware). TVM's value proposition is strongest when your deployment targets include non-NVIDIA hardware or when you need a single compilation stack across diverse platforms.

## Glossary & Acronyms

| Term | Full Form | Definition | Why It Matters |
|------|-----------|------------|----------------|
| TVM | Tensor Virtual Machine | Open-source end-to-end deep learning compiler optimizing models for diverse hardware through graph-level and operator-level transformations with learning-based auto-tuning [1] | The subject of this report — the foundational ML compiler enabling performance portability across CPUs, GPUs, FPGAs, and accelerators |
| Relay | N/A (TVM high-level IR) | TVM's established high-level functional IR for neural networks, supporting control flow, recursion, and graph-level optimizations [6] | The workhorse IR powering TVM deployments for years, now being gradually superseded by Relax |
| Relax | Relay Next | TVM's next-gen graph IR with first-class symbolic shape annotations for dynamic shape support and cross-level abstraction [2] | Represents TVM's future; enables dynamic shape workloads critical for LLMs with up to 27% latency reduction |
| TIR | Tensor Intermediate Representation | TVM's low-level IR operating at the loop-nest level with multi-dimensional load/store, threading, and vector/tensor instructions [6] | The layer where hardware-specific optimizations happen — determines final performance on target hardware |
| TVM Unity | N/A | TVM's design vision for cross-level optimization, where graph-level (Relax) and loop-level (TIR) abstractions co-exist and co-optimize in a single IRModule [8] | Architectural philosophy driving Relax's development and TVM's differentiator vs. MLIR's decoupled model |
| MLIR | Multi-Level Intermediate Representation | LLVM subproject providing extensible compiler infrastructure through a dialect-based design [4] | The competing/complementary infrastructure to TVM, foundation of the OpenXLA ecosystem |
| IRModule | Intermediate Representation Module | TVM's core compilation unit containing both Relax functions and TIR primitive functions [6] | Central data structure — understanding it is essential for working with TVM's optimization pipeline |
| MetaSchedule | N/A | TVM's 3rd-generation unified auto-tuning system combining template-based and template-free approaches [7] | Auto-tuning quality directly determines TVM output performance |
| AutoTVM | N/A | TVM's 1st-generation template-based auto-tuning system [1] | Historical context for understanding TVM's auto-tuning evolution |
| Ansor | N/A (also AutoScheduler) | TVM's 2nd-generation template-free auto-scheduler using derivation-based sketch generation [3] | Eliminated need for hand-written templates, democratizing TVM optimization |
| DLight | N/A | Pre-defined lightweight TIR schedules optimized for LLM GPU workloads with dynamic shape support [6] | Enables fast LLM optimization without expensive auto-tuning search |
| StableHLO | Stable High-Level Operations | Backward-compatible ML compute opset serving as a portability layer in the OpenXLA ecosystem [17] | Emerging universal exchange format potentially bridging TVM and MLIR ecosystems |
| HLO/MHLO | High-Level Operations / Meta HLO | XLA's internal opset; StableHLO is its backward-compatible successor designed for cross-framework portability [17] | Historical context for StableHLO's origins |
| XLA | Accelerated Linear Algebra | Google's ML compiler for TensorFlow/JAX, now part of OpenXLA [12] | Primary competitor for JAX/TPU workloads |
| IREE | Intermediate Representation Execution Environment | MLIR-based ML execution runtime, part of OpenXLA [16] | Closest MLIR-ecosystem analog to TVM's deployment story |
| torch-mlir | N/A | LLVM Incubator project bridging PyTorch to MLIR compiler infrastructure [11] | Alternative to TVM's PyTorch frontend for the MLIR compilation path |
| TorchDynamo | N/A | PyTorch's graph capture frontend for torch.compile [9] | Entry point for PyTorch compilation — both TVM and TorchInductor receive its output |
| TorchInductor | N/A | Default torch.compile backend generating optimized code via Triton (GPU) and C++/OpenMP (CPU) [9] | TVM's primary competitor in PyTorch compilation for NVIDIA GPUs |
| Triton | N/A (OpenAI Triton) | GPU programming language and compiler used by TorchInductor to generate optimized CUDA/ROCm kernels [9] | The technology underlying TorchInductor's NVIDIA GPU performance advantage |
| AOTAutograd | Ahead-of-Time Autograd | PyTorch component tracing the backward pass ahead of time in the torch.compile pipeline [9] | Enables compilation of training workloads, not just inference |
| BYOC | Bring Your Own Codegen | TVM mechanism for integrating custom backend code generators [6] | Key extensibility mechanism for supporting diverse hardware without core changes |
| TOPI | Tensor Operator Inventory | TVM's library of commonly used tensor operator templates [6] | Provides the starting point for auto-tuning |
| TVMScript | N/A | Python-based DSL for writing and inspecting TVM programs [6] | Primary developer interface for custom TVM optimizations |
| TOSA | Tensor Operator Set Architecture | Hardware-agnostic tensor operation specification, an MLIR dialect [11] | Represents push toward hardware-agnostic ML compilation |
| OpenXLA | N/A | Industry consortium providing modular ML compiler toolchain built on MLIR [12] | Most significant organized effort to standardize ML compilation infrastructure |
| PJRT | Plugin-based Just-in-time Runtime | Hardware-independent interface within OpenXLA [12] | OpenXLA's approach to hardware portability, analogous to TVM's BYOC |
| MLC-LLM | Machine Learning Compilation for LLMs | Universal LLM deployment engine built on TVM Unity [13] | Most visible production application of modern TVM |
| FX Graph | N/A | PyTorch's intermediate graph representation used as input for torch.compile backends [9] | Common input format enabling TVM and other compilers to consume PyTorch models |
| PagedKVCache | Paged Key-Value Cache | TVM's paged attention KV caching for efficient LLM inference [13] | Critical for long-context LLM inference (32K–128K tokens) |
| OctoML | N/A | Commercial company founded by TVM's original creators to provide TVM-based optimization-as-a-service; acquired by NVIDIA in 2024 [15] | Key indicator of TVM's commercial trajectory and future governance questions |
| SoC | System on a Chip | Integrated circuit combining CPU, GPU, NPU, memory controller, and other components on a single die (e.g., Qualcomm Snapdragon, Apple M-series) | Deployment target category where TVM's cross-platform portability is most valuable |
| Shardy | N/A | OpenXLA component providing tensor partitioning infrastructure for distributed ML workloads [12] | Part of the OpenXLA ecosystem relevant to large-scale distributed inference/training |
| NCHW / NHWC | N=batch, C=channels, H=height, W=width | Two common data layouts for 4D feature map tensors in CNNs; NCHW is preferred by NVIDIA CUDA, NHWC by ARM and many mobile accelerators | TVM's layout transformation pass automatically selects the optimal layout for the target hardware |

## How Things Relate (Concept Map)

- **TVM** contains **Relay** and **Relax** as high-level graph IRs, and **TIR** as its low-level loop IR — all wrapped in the **IRModule** compilation unit
- **Relax** extends and supersedes **Relay**, adding dynamic shape support and cross-level abstraction as part of the **TVM Unity** vision; TVM Unity is the overarching design philosophy realized through Relax
- **MetaSchedule** improves upon **Ansor**, which itself improved upon **AutoTVM** — three generations of auto-tuning with increasing automation
- **TorchDynamo** captures PyTorch graphs that can be compiled by either **TorchInductor** (default) or **TVM** (alternative backend); **AOTAutograd** extends TorchDynamo's captured forward graph with backward-pass tracing
- **torch-mlir** competes with TVM's PyTorch frontend by bridging PyTorch to **MLIR** dialects, ultimately producing **StableHLO** for **XLA** or **IREE**
- **IREE** competes with **TVM** for ML deployment (especially edge/mobile), but builds on **MLIR** infrastructure rather than TVM's custom IR stack
- **OpenXLA** builds on **MLIR** and includes **XLA**, **StableHLO**, **IREE**, **Shardy**, and **PJRT** as components
- **MLC-LLM** builds on **TVM** (Relax + TIR) to deliver cross-platform LLM deployment; **DLight** provides pre-defined **TIR** schedules that bypass auto-tuning for fast LLM optimization
- **BYOC** enables **TVM** to integrate custom hardware backends without core modifications, making TVM accessible to ASIC and FPGA vendors
- **StableHLO** is emerging as a potential bridge between the TVM and MLIR ecosystems, since both can produce and/or consume it
- **OctoML** was the commercial arm of TVM, now acquired by **NVIDIA**, creating open questions about TVM's commercial support trajectory

## References

[1] T. Chen, T. Moreau, Z. Jiang, L. Zheng et al., "TVM: An Automated End-to-End Optimizing Compiler for Deep Learning," in *Proc. OSDI '18*, 2018. [Online]. Available: https://arxiv.org/abs/1802.04799

[2] R. Lai et al., "Relax: Composable Abstractions for End-to-End Dynamic Machine Learning," arXiv:2311.02103, 2023. [Online]. Available: https://arxiv.org/abs/2311.02103

[3] L. Zheng et al., "Ansor: Generating High-Performance Tensor Programs for Deep Learning," in *Proc. OSDI '20*, 2020. [Online]. Available: https://arxiv.org/abs/2006.06762

[4] C. Lattner, M. Amini et al., "MLIR: Scaling Compiler Infrastructure for Domain Specific Computation," in *Proc. CGO '21*, 2021. [Online]. Available: https://ieeexplore.ieee.org/document/9370308

[5] C. Lattner, J. Pienaar et al., "MLIR: A Compiler Infrastructure for the End of Moore's Law," arXiv:2002.11054, 2020. [Online]. Available: https://arxiv.org/abs/2002.11054

[6] Apache TVM Project, "TVM Design and Architecture Documentation," tvm.apache.org, 2024. [Online]. Available: https://tvm.apache.org/docs/arch/index.html

[7] Apache Software Foundation, "Apache TVM GitHub Repository and Release Notes," GitHub, 2025. [Online]. Available: https://github.com/apache/tvm

[8] Apache TVM Project, "TVM Unity: The Next Leap in Machine Learning Compilation," tvm.apache.org, 2021. [Online]. Available: https://tvm.apache.org/2021/12/15/tvm-unity

[9] PyTorch Team, "torch.compile and TorchInductor Documentation," pytorch.org, 2025. [Online]. Available: https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler.html

[10] Modular, "Democratizing AI Compute Part 6: What about AI Compilers?" modular.com, 2024. [Online]. Available: https://www.modular.com/blog/democratizing-ai-compute-part-6-what-about-ai-compilers

[11] LLVM Project, "torch-mlir: Bridge between PyTorch and MLIR," GitHub, 2024. [Online]. Available: https://github.com/llvm/torch-mlir

[12] OpenXLA, "OpenXLA Project," openxla.org, 2025. [Online]. Available: https://openxla.org/

[13] MLC AI, "MLC-LLM: Universal LLM Deployment Engine," GitHub, 2024. [Online]. Available: https://github.com/mlc-ai/mlc-llm

[14] HuggingFace, "HuggingFace Optimum Documentation," huggingface.co, 2025. [Online]. Available: https://huggingface.co/docs/optimum

[15] Various, "OctoAI acquired by NVIDIA," Industry reports, 2024. [Online]. Available: https://blogs.nvidia.com/blog/octoai/

[16] IREE Organization, "IREE: Intermediate Representation Execution Environment," GitHub, 2024. [Online]. Available: https://github.com/iree-org/iree

[17] OpenXLA, "StableHLO Specification," GitHub, 2024. [Online]. Available: https://github.com/openxla/stablehlo

[18] AWS, "Amazon SageMaker Neo," aws.amazon.com, 2025. [Online]. Available: https://aws.amazon.com/sagemaker/neo/

[19] Various, "The Deep Learning Compiler: A Comprehensive Survey," arXiv:2002.03794, 2020. [Online]. Available: https://arxiv.org/abs/2002.03794

[20] Various, "Autotuning Apache TVM-based Scientific Applications Using Bayesian Optimization," arXiv:2309.07235, 2023. [Online]. Available: https://arxiv.org/abs/2309.07235

[21] Intel, "OpenVINO 2025 Documentation," intel.com, 2025. [Online]. Available: https://docs.openvino.ai/

[22] PyTorch Team, "ExecuTorch Documentation," pytorch.org, 2025. [Online]. Available: https://pytorch.org/executorch/

[23] NVIDIA, "TensorRT-LLM," nvidia.com, 2025. [Online]. Available: https://developer.nvidia.com/tensorrt

[24] Apache TVM Project, "TVM Conference Talks and Community Updates," tvm.apache.org, 2024. [Online]. Available: https://tvm.apache.org/

[25] torch-mlir contributors, "torch-mlir FOSDEM 2025 Presentation," FOSDEM, 2025. [Online]. Available: https://fosdem.org/2025/
