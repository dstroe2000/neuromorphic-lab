# AI Compiler Stack and ML Deployment Pipeline: Technical Deep-Dive

## Key Questions This Report Answers

1. How does MLIR's extensible dialect system enable compilation from high-level ML frameworks to diverse hardware targets, including neuromorphic accelerators?
2. What are the practical workflows for adding custom hardware backends to IREE or TVM compilers?
3. How do Triton's block-level programming and XLA's whole-graph compilation differ in their approaches to kernel generation?
4. What is Intel's strategy for heterogeneous computing with oneAPI/SYCL, and how might it integrate with neuromorphic hardware like Loihi?
5. What does an end-to-end AI model optimization pipeline look like, from training to deployment?
6. Which CUDA programming concepts transfer to neuromorphic computing, and what fundamentally differs?
7. Why doesn't a production-grade compiler stack exist yet for neuromorphic hardware, and what would it take to build one?

## The Mental Model (How to Think About This)

Think of modern AI compilation as a series of translators working together. At the top, you write your neural network in PyTorch or JAX—high-level frameworks designed for human productivity. At the bottom, you have wildly different hardware: NVIDIA GPUs with CUDA cores, Intel CPUs with AVX-512 instructions, Google TPUs with systolic arrays, or neuromorphic chips with asynchronous spiking neurons.

The compiler stack bridges this gap through **progressive lowering**: your high-level "train this network" gradually transforms into "execute these specific operations on this specific hardware." MLIR (Multi-Level Intermediate Representation) provides the infrastructure for this translation. It's like a construction set where you can build custom translation layers (called "dialects") for any hardware or abstraction level. Triton lets you write GPU kernels in Python instead of CUDA C. XLA optimizes entire computation graphs automatically. OpenVINO squeezes maximum performance from Intel hardware through quantization and fusion.

But here's the gap: neuromorphic hardware—chips that compute with brain-inspired spiking neurons rather than matrix multiplies—doesn't yet have this polished toolchain. There's no "pip install neuromorphic-compiler" that bridges PyTorch to Loihi the way torch.compile bridges PyTorch to GPUs. Building this bridge is the frontier challenge.

## Prerequisites / What You Need to Know First

This report assumes familiarity with neural networks (forward/backward propagation, layers, training), basic understanding of GPUs and parallel computing, and some exposure to compilers (what an intermediate representation is, why compilation has stages). If you've trained a PyTorch model and wondered how it actually runs on a GPU, you have enough background. We'll build up from there to compiler internals, hardware-specific optimizations, and the unique challenges of neuromorphic computing. No expertise in compiler theory required—we'll explain the key concepts as we go.

## TL;DR

Modern AI compilation stacks like MLIR, Triton, XLA, IREE, and OpenVINO enable portable, high-performance model deployment across CPUs, GPUs, and TPUs through progressive lowering and hardware-specific optimization. MLIR's extensible dialect system is the common foundation, allowing custom compilation paths for new hardware. Intel's oneAPI/SYCL provides cross-platform heterogeneous computing via C++, while OpenVINO optimizes inference for Intel hardware through aggressive quantization and fusion. However, neuromorphic hardware (event-driven, spike-based, asynchronous) remains separated from these mainstream stacks—no production MLIR-based neuromorphic compiler exists. Bridging this gap requires designing custom MLIR dialects for spike-based operations, event-driven scheduling, and sparse temporal dynamics, then integrating with backends like Intel's Lava/Loihi ecosystem.

## 1. What Problem Does This Solve?

Deep learning's explosive growth created a hardware diversity crisis. NVIDIA GPUs dominated early on, but now we have Google TPUs optimized for matrix multiplies, Intel CPUs with specialized AI instructions (AMX, AVX-512), Apple Neural Engines for mobile, and emerging neuromorphic chips that compute with asynchronous spikes rather than synchronous arithmetic [6,7]. Each hardware platform has unique programming models, memory hierarchies, and optimization strategies.

The traditional approach—write framework code once (PyTorch/TensorFlow), let vendors provide kernels—breaks down at scale. Custom kernels are expensive to develop and maintain. Hardware-specific code locks you into one vendor. Performance portability becomes impossible. You can't easily experiment with new accelerators because porting an entire ML framework takes years.

This is where the modern compiler stack comes in. Rather than framework developers writing CUDA kernels, GPU kernels, TPU kernels, and neuromorphic kernels separately, we build **compiler infrastructure** that can target any hardware from a common intermediate representation [1,9]. MLIR provides the extensible foundation. Triton makes GPU kernel writing accessible via Python [2]. XLA automatically optimizes computation graphs for any backend. IREE provides a runtime that works from edge to cloud [3]. OpenVINO squeezes maximum performance from Intel hardware with aggressive quantization [5].

The goal: **write your model once, compile it to any hardware efficiently, without sacrificing performance or locking into one vendor's ecosystem**. For Intel specifically, this means bridging traditional data-parallel AI (oneAPI, OpenVINO) with neuromorphic research hardware (Loihi/Lava)—a gap that remains unsolved [6,7,9].

## 2. How It Works

The modern AI compiler stack has several layers working together. This section walks through each component and how they fit together.

### 2.1 MLIR: The Universal Foundation

MLIR (Multi-Level Intermediate Representation) is the infrastructure layer that makes everything else possible [1,11]. Think of it as LEGO for compilers: it provides the building blocks (operations, types, attributes), the rules for combining them (dialects, passes), and the infrastructure for transforming code through multiple abstraction levels (progressive lowering).

The key innovation is **dialects**—namespaced collections of operations tailored to specific domains. The Linalg dialect represents linear algebra (convolutions, matrix multiplies) at a high level, independent of whether you're targeting a CPU, GPU, or accelerator [1]. The GPU dialect models GPU-specific abstractions like thread blocks and shared memory. The LLVM dialect is the final lowering target before code generation.

A typical compilation flow looks like:
```
PyTorch model → Torch-MLIR → Linalg-on-Tensors → Linalg-on-Buffers →
SCF (loops) → GPU/Vector/CPU dialects → LLVM IR → Machine Code
```

At each stage, you preserve semantic meaning while exposing more hardware-specific details. High-level optimizations (fusing convolution+batch_norm+relu) happen in Linalg. Memory layout choices happen when converting tensors to buffers. Loop tiling and vectorization happen in SCF/Affine. GPU-specific thread mapping happens in the GPU dialect [1,11,13].

This progressive lowering is what enables **customization**. Want to add a neuromorphic accelerator? Define a new dialect with spike-propagation and neuron-update operations, write transformations from Linalg to your dialect, and provide a code generator to your hardware. Intel did exactly this with the XeVM dialect for Intel GPUs—upstreamed to mainline LLVM in 2025 [8,19]. This demonstrates that the MLIR infrastructure is mature enough to support vendor-specific extensions while maintaining compatibility with the broader ecosystem.

### 2.2 Triton: Python-Native GPU Kernel Programming

Writing CUDA kernels is hard. You think in terms of individual threads, manage shared memory manually, worry about memory coalescing, synchronize with barriers, and tune occupancy by hand. Triton changes the paradigm: you write kernels at the **block level**, thinking in terms of tiles of data rather than individual elements [2,17,18,19].

Here's the mental shift. In CUDA, you write: "Thread (x, y, z) in block (bx, by, bz) loads element A[...]." In Triton, you write: "Load this 128×128 tile of A into shared memory, compute on the tile, write the tile back." Triton's compiler handles thread mapping, memory coalescing, and synchronization automatically.

A competitive FP16 matrix multiply can be written in approximately 25 lines of Triton code, matching cuBLAS performance. The programmer specifies tile sizes, memory access patterns, and computation logic. The Triton compiler (now built on MLIR) handles the tedious low-level details [2,17].

The compilation pipeline:
```
Python AST → Triton-IR (block-level, SSA form) → Triton-GPUIR (thread mapping) →
LLVM IR → PTX (NVIDIA) / GCN (AMD) / SPIR-V (Intel, experimental)
```

Triton's killer feature is **auto-tuning**: programs expose tuning parameters (tile sizes, loop ordering), and the runtime searches for optimal configurations on your specific hardware [2]. This is why PyTorch's torch.compile now uses Triton (via TorchInductor) as its default kernel generator—it combines ease of use with near-hand-tuned performance.

Known limitations: premature lowering from block level directly to thread level (according to the ML-Triton paper [17]), layout-related bugs (12% of GitHub issues), and limited multi-GPU communication [17,18]. But for single-kernel optimization, it's become the standard.

### 2.3 XLA: Whole-Graph Compilation

XLA (Accelerated Linear Algebra) takes the opposite approach from Triton: instead of explicit kernel programming, you write normal JAX or TensorFlow code, and XLA compiles **entire computation graphs** into optimized machine code [21].

The key abstraction is HLO (High-Level Optimizer)—a graph representation where nodes are operations (conv, matmul, elementwise functions) and edges are tensors [9]. XLA makes aggressive graph-level optimizations:

- **Fusion**: Combine multiple operations into single kernels to eliminate memory round-trips. Example: conv → batch_norm → relu becomes one fused kernel. DNNFusion research shows 8.8× more fusion opportunities and 9.3× speedup over baseline [14].
- **Layout assignment**: Choose optimal memory layouts (NCHW vs NHWC) for your target hardware.
- **Buffer assignment**: Reuse memory buffers to minimize peak memory usage.
- **Scheduling**: Determine the optimal execution order to maximize parallelism and minimize synchronization.

XLA targets CPUs (via LLVM), GPUs (via LLVM+CUDA/ROCm), and TPUs (via custom backend). The programmer writes high-level code; XLA handles all optimization [21].

The trade-off: less control than Triton. If XLA's heuristics don't fuse your operations optimally, you can't easily override it. But for most workloads, the automatic optimization is good enough—and you get portability across hardware for free.

### 2.4 IREE: MLIR-Native Runtime

IREE (Intermediate Representation Execution Environment) is Google's answer to "how do we deploy MLIR-compiled models everywhere?" [3]. Unlike XLA (primarily cloud TPUs/GPUs) or Triton (GPU kernels), IREE targets **edge to cloud** with a hardware abstraction layer (HAL).

The architecture:
```
ML Model → StableHLO → IREE Compiler → HAL Module → Target Device
                                             ↓
                            CPU / GPU / Custom Accelerator (via PluginAPI)
```

IREE compiles models ahead-of-time into optimized bytecode. The HAL provides a portable execution interface: allocate buffers, dispatch kernels, synchronize. Backend developers implement this interface for their hardware [3].

For custom accelerators (like Loihi), IREE's PluginAPI is the extension point. You provide:
1. HAL device interface implementation (memory allocation, kernel dispatch)
2. Code generation pipeline from IREE's intermediate dialects to your hardware
3. Runtime integration for synchronization and memory management

No one has built an IREE-Loihi backend yet, but the architecture supports it [3,31].

### 2.5 OpenVINO: Intel's Inference Optimizer

OpenVINO focuses on one thing: **maximum inference performance on Intel hardware** [5]. It's not a research framework—it's a deployment toolkit that takes trained models (PyTorch/TensorFlow/ONNX) and optimizes them aggressively for Intel CPUs, GPUs, and NPUs.

The pipeline:
```
Trained Model → Model Optimizer (graph transformations) → Intermediate Representation (IR) →
Inference Engine → CPU/GPU/NPU/AUTO plugin
```

The secret sauce is NNCF (Neural Network Compression Framework) [5]:
- **Post-Training Quantization (PTQ)**: Convert FP32 → INT8 without retraining, using calibration data to minimize accuracy loss. Accuracy-aware mode searches for optimal per-layer bit-widths.
- **Quantization-Aware Training (QAT)**: Fine-tune with fake quantization operators for better accuracy at low bit-widths.
- **Structured pruning**: Remove entire channels or layers based on importance scores.
- **Mixed precision**: Selectively use FP16/INT8 per layer, balancing accuracy and speed.

OpenVINO's quantization toolkit is more mature than torch.compile or JAX—it's built for production deployment where model size and latency matter [5,22,24]. Recent work like QuantuneV2 shows compiler-level mixed precision at the IR level, achieving 10.28% accuracy improvement and 12.52% speed increase [22].

The 2025.4 release focuses on GenAI: LLM serving, vision-language models, KServe/TensorFlow Serving compatibility, and Intel Core Ultra NPU support [5]. For Intel-specific deployments, OpenVINO is the clear choice.

### 2.6 oneAPI/SYCL: Cross-Platform Heterogeneous Computing

While OpenVINO targets AI inference, oneAPI is Intel's broader strategy for heterogeneous computing across **any** workload—AI, HPC, data analytics [4]. The centerpiece is SYCL, an open standard (Khronos Group) for single-source C++ programming across CPUs, GPUs, FPGAs, and accelerators.

The programming model:
```cpp
queue q; // Device queue
buffer<float> buf(data, range<1>(N)); // Data buffer
q.submit([&](handler& h) {
  accessor a(buf, h); // Access pattern
  h.parallel_for(range<1>(N), [=](id<1> i) {
    a[i] *= 2.0f; // Computation
  });
});
```

Host and device code in one file. The compiler (DPC++, Intel's LLVM-based SYCL implementation) handles code generation for your target [4]. USM (Unified Shared Memory) provides automatic memory management across devices—no manual copies.

The ecosystem includes:
- **oneMKL**: Optimized math kernels
- **oneDNN**: Deep neural network primitives (used by PyTorch and TensorFlow for Intel acceleration)
- **oneDAL**: Data analytics libraries
- **oneCCL**: Collective communications for distributed training

The catch: SYCL maturity lags CUDA. NVIDIA support exists (via DPC++ with CUDA backend), but performance isn't always competitive [4,10]. For cross-vendor portability, SYCL is the right abstraction. For peak NVIDIA GPU performance, CUDA still reigns.

## 3. Under the Hood

This section dives into the key technical mechanisms that make these systems work.

### 3.1 Progressive Lowering in MLIR

The genius of MLIR is its flexible lowering strategy. Instead of a single giant compilation step (like LLVM), you have multiple stages, each with its own optimization opportunities [1,11].

Example: compiling a PyTorch convolution to x86 assembly.

**Stage 1: Torch-MLIR**
```mlir
%result = torch.aten.conv2d %input, %weight, %bias, ...
```
Still PyTorch semantics. Operations have dynamic shapes, Python-like behavior.

**Stage 2: Linalg-on-Tensors**
```mlir
%result = linalg.conv_2d_nhwc_hwcf
  ins(%input, %weight : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>)
  outs(%output : tensor<?x?x?x?xf32>)
```
Now it's pure tensor algebra, no framework-specific semantics. Shapes still dynamic.

**Stage 3: Linalg-on-Buffers (after bufferization)**
```mlir
linalg.conv_2d_nhwc_hwcf
  ins(%input_memref, %weight_memref : memref<?x?x?x?xf32>, memref<?x?x?x?xf32>)
  outs(%output_memref : memref<?x?x?x?xf32>)
```
Memory allocated, buffer reuse decided. Still hardware-agnostic.

**Stage 4: SCF (after tiling)**
```mlir
scf.for %i = %c0 to %N step %tile_size {
  scf.for %j = %c0 to %M step %tile_size {
    // Tiled computation on memrefs
  }
}
```
Loops exposed for optimization—tiling, unrolling, parallelization.

**Stage 5: Vector dialect**
```mlir
%vec = vector.load %memref[%i] : memref<?xf32>, vector<8xf32>
%result = vector.fma %vec, %other, %acc : vector<8xf32>
```
SIMD operations for CPU vectorization (AVX2/AVX-512).

**Stage 6: LLVM dialect → LLVM IR → x86 assembly**

Each stage enables specific optimizations: operator fusion at Linalg level, memory layout at bufferization, loop optimization at SCF, SIMD at Vector. This is why MLIR compilers can match or beat hand-tuned code—they optimize at every level [1,11,13].

### 3.2 Triton's Block-Level Abstraction

Triton's core innovation is hiding the thread/block/warp complexity behind a block-oriented programming model [2,17]. The programmer writes:

```python
@triton.jit
def matmul_kernel(A, B, C, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Block indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Load BLOCK_M x BLOCK_K tile of A
    a_tile = tl.load(A + offsets_a, mask=mask_a)
    # Load BLOCK_K x BLOCK_N tile of B
    b_tile = tl.load(B + offsets_b, mask=mask_b)

    # Compute tile of C
    c_tile = tl.dot(a_tile, b_tile)

    # Store result
    tl.store(C + offsets_c, c_tile, mask=mask_c)
```

The Triton compiler transforms this into CUDA PTX by:
1. **Thread mapping**: Each tile operation becomes a grid of threads cooperating to load/compute/store the tile.
2. **Memory optimization**: Automatically allocate shared memory for tiles, avoid bank conflicts, coalesce global memory accesses.
3. **Synchronization**: Insert barriers between load/compute/store phases.

The programmer controls performance-critical decisions (tile size, loop structure) without low-level CUDA boilerplate [2,17]. Auto-tuning searches over tile sizes to find the optimal configuration.

Recent research identifies issues: premature lowering loses optimization opportunities (according to ML-Triton [17]), layout bugs plague 12% of issues (according to the Linear Layouts paper [19]), multi-GPU communication needs explicit management (according to Iris [18]). But the productivity gains are massive—vLLM, DeepSpeed, and PyTorch all adopted Triton for custom kernels.

### 3.3 Operator Fusion Deep-Dive

Fusion is the single most impactful compiler optimization for ML workloads [14,16,20]. Every memory round-trip (write intermediate result to DRAM, read it back) costs orders of magnitude more than arithmetic. Fusing operations eliminates these round-trips.

**Simple fusion**: conv → batch_norm → relu
```
# Unfused (3 kernels, 2 DRAM round-trips)
tmp1 = conv(input, weight)  # Write tmp1 to DRAM
tmp2 = batch_norm(tmp1)     # Read tmp1, write tmp2
output = relu(tmp2)         # Read tmp2, write output

# Fused (1 kernel, 0 intermediate DRAM)
output = relu(batch_norm(conv(input, weight)))
```

**Advanced fusion** (Neptune, arXiv:2510.08726 [20]): Fuse across reduction boundaries and complex control flow. Example: softmax involves exp, sum, and divide—normally 3 kernels. Neptune fuses them into one, eliminating intermediate writes.

DNNFusion research shows 8.8× more fusion opportunities and 9.3× speedup by using mathematical properties beyond simple pattern matching [14]. The key: recognize that conv+bn is mathematically equivalent to a single conv with modified weights, so you can fold batch_norm into the conv entirely.

OpenVINO's graph compiler (oneDNN Graph) uses hybrid fusion: pattern-based for common subgraphs, mathematical-property-based for flexible cases [15,16].

### 3.4 Quantization: From FP32 to INT8/INT4

Quantization reduces model size and speeds up inference by using lower-precision arithmetic [5,22,23,24,25]. The challenge: maintain accuracy while mapping FP32's wide dynamic range to INT8's 256 values.

**Post-Training Quantization (PTQ)**: Given a trained FP32 model, collect activation statistics on calibration data, then compute scale factors for each layer:
```
quantized_value = round(float_value / scale)
dequantized_value = quantized_value * scale
```

Symmetric quantization uses scale only. Asymmetric adds a zero-point offset. Per-tensor quantization uses one scale for the entire tensor. Per-channel quantization (used by NNCF) uses different scales per output channel, improving accuracy [5].

**Quantization-Aware Training (QAT)**: Insert fake quantization operators during training:
```python
def fake_quantize(x, scale, bits=8):
    qmin, qmax = -2**(bits-1), 2**(bits-1) - 1
    x_q = torch.clamp(torch.round(x / scale), qmin, qmax)
    return x_q * scale  # Dequantized for gradient flow
```

The network learns to be robust to quantization noise, achieving better accuracy than PTQ [5].

**Mixed precision**: Not all layers need the same bit-width. QuantuneV2 (2025) does compiler-level mixed precision selection at the IR level, achieving 10.28% accuracy improvement and 12.52% speed increase over uniform INT8 [22]. TorchAO unifies training and serving quantization, supporting INT4, INT8, FP8, and exotic formats like MXFP4 (microscaling) [24,25].

State-of-the-art (2025): Progressive Mixed-Precision Decoding (PMPD) for LLM inference—use higher precision early in generation (when accuracy matters), lower precision later (when continuation is more constrained)—achieves 1.4-12.2× speedup [23].

## 4. Performance & Benchmarks

Real-world compiler performance varies dramatically based on workload, hardware, and optimization effort.

**MLIR-based compilers**: The Transform Dialect (CGO 2025) enables 10× speedup over C-level optimization for DSP workloads by exposing fine-grained control over tiling, vectorization, and parallelization [11,12].

**Triton**: A well-tuned Triton kernel matches hand-written CUDA for matrix multiply (within 5% of cuBLAS). According to vLLM developers [2,18], custom Triton kernels for attention produce 14× throughput improvement for LLM serving.

**XLA**: On TPUs, XLA-compiled JAX code runs 2-5× faster than eager mode due to fusion and layout optimization. On GPUs, gains are smaller (1.2-2×) because vendor libraries (cuDNN, cuBLAS) are already highly optimized [21].

**OpenVINO quantization**: INT8 models typically run 2-4× faster than FP32 on Intel CPUs with VNNI instructions. Accuracy loss is typically <1% for CNNs, 1-3% for transformers with PTQ, <0.5% with QAT [5,22].

**Fusion impact**: DNNFusion achieves 9.3× speedup on ResNet-50 by discovering 8.8× more fusion opportunities than TensorFlow XLA or TVM [14]. Neptune (2025) shows 1.5-2.5× speedup by fusing across reduction boundaries [20].

Trade-offs:
- **Compilation time**: MLIR-based compilers are slower to compile than eager execution (seconds to minutes vs milliseconds). Use ahead-of-time compilation for deployment.
- **Debuggability**: Fused kernels and quantized models are harder to debug than FP32 eager mode.
- **Portability vs peak performance**: Cross-platform code (SYCL, MLIR) achieves 70-90% of hand-tuned performance. For peak performance, use vendor libraries (cuDNN on NVIDIA, oneDNN on Intel).

## 5. Practical Considerations

Deploying these compiler stacks in production involves several gotchas.

### Numerical Precision

Compiler optimizations can change numerical results. Fusion reorders floating-point operations, which are non-associative. Quantization introduces rounding error. Always validate accuracy on representative data, not just training set.

### Memory Management

Compiler-generated code may have different memory allocation patterns than eager execution. IREE uses arena allocation for predictable memory usage. Triton relies on shared memory auto-management, which can fail for complex access patterns [3,18].

### Dynamic Shapes

Many compilers assume static shapes for optimization. Dynamic shapes (variable batch size, sequence length) force conservative code generation or runtime recompilation. XLA caches compiled functions per shape. torch.compile supports dynamic shapes via symbolic tracing but may recompile frequently [21].

### Hardware-Specific Features

To get peak performance, you need hardware-specific optimizations: Tensor Cores on NVIDIA GPUs, AMX on Intel Sapphire Rapids, WMMA on AMD. These require explicit lowering in the compiler stack. Generic MLIR lowering won't automatically use them—you need target-specific passes [1,8,19].

### Integration with Existing Code

Gradual adoption is key. You don't rewrite your entire codebase to use a new compiler. Start with performance-critical kernels (attention, custom layers). Use Triton for GPU kernels, OpenVINO for Intel inference, torch.compile for general PyTorch acceleration [2,5,24].

### Debugging and Profiling

When a compiled kernel is slower than expected, you need tools:
- **MLIR**: `--mlir-print-ir-after-all` dumps IR after each pass
- **Triton**: `triton-vis` visualizes thread mapping and memory access patterns
- **XLA**: `XLA_FLAGS=--xla_dump_to=/tmp/xla` dumps HLO graphs
- **OpenVINO**: Benchmark app for layer-by-layer profiling

Learn to read IR and use profilers (nsys for NVIDIA, VTune for Intel) to understand where time is spent [1,2,5].

## 6. Implementation Roadmap

Here's how to actually build expertise and deploy these stacks.

### Phase 1: Foundations (2-4 weeks)

**Goal**: Understand the basic abstractions and workflows.

1. **MLIR tutorials**: Work through MLIR's Toy tutorial (build a custom dialect, write lowering passes). Understand operation definition, type systems, and pass infrastructure [1].
2. **Triton**: Write a simple GPU kernel (element-wise operation, then matrix multiply). Benchmark against PyTorch eager mode. Use auto-tuning [2].
3. **SYCL**: Port a CUDA kernel to SYCL using DPC++. Compare performance and code complexity [4].

**Deliverable**: A simple conv → relu fusion implemented as MLIR pass, a Triton matrix multiply kernel, a SYCL vector add.

### Phase 2: Integration (4-8 weeks)

**Goal**: Connect the pieces—compile a real model end-to-end.

1. **Torch-MLIR**: Convert a PyTorch model (ResNet-18) to MLIR via Torch-MLIR. Inspect the generated Linalg IR [1].
2. **OpenVINO**: Quantize the same model with NNCF (PTQ), deploy with the inference engine, benchmark CPU/GPU performance [5].
3. **IREE**: Compile a simple model to IREE bytecode, run on CPU/GPU via HAL [3].

**Deliverable**: Comparative benchmark of the same model on PyTorch eager, torch.compile, OpenVINO INT8, IREE.

### Phase 3: Custom Backend (8-12 weeks)

**Goal**: Extend the stack for a new accelerator (or neuromorphic hardware).

1. **Define MLIR dialect**: Create a custom dialect for your target hardware. For neuromorphic: spike propagation ops, neuron state update, synaptic plasticity [1,31].
2. **Write lowering passes**: Convert Linalg or StableHLO to your dialect. Handle memory allocation, scheduling, and synchronization [1,13].
3. **Implement backend**: Either IREE PluginAPI (for runtime integration) or standalone code generator [3].
4. **Benchmarking**: Compare against naive mappings, measure hardware utilization, latency, energy.

**Deliverable**: A working prototype that compiles a simple SNN (Spiking Neural Network) model from PyTorch (via snnTorch) to your target hardware via MLIR.

## 7. Getting Started

**Immediate next steps** (for someone targeting the Intel AI Software Architect role):

1. **Clone and build llvm-project** with MLIR enabled. Run the Toy tutorial to build intuition for dialects and passes [1].
   ```bash
   git clone https://github.com/llvm/llvm-project.git
   cmake -G Ninja ../llvm -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_TARGETS_TO_BUILD="X86"
   ninja check-mlir
   ```

2. **Install and experiment with Triton**. Write a simple kernel (element-wise), then matmul. Compare to PyTorch eager and torch.compile [2].
   ```bash
   pip install triton
   # Follow Triton tutorials at triton-lang.org
   ```

3. **Set up OpenVINO**. Quantize a pre-trained model (ResNet-50 from torchvision), benchmark FP32 vs INT8 on your Intel CPU [5].
   ```bash
   pip install openvino-dev nncf
   # Follow OpenVINO quickstart guides
   ```

4. **Read Intel Lava documentation** and understand the Loihi 2 architecture. Identify the gap: how would you compile a PyTorch SNN (snnTorch) to Lava? Where does MLIR fit in? [6,7,30].

5. **Explore NIR (Neuromorphic Intermediate Representation)** as a potential interoperability layer. It's the ONNX equivalent for neuromorphic, supporting 11 platforms [30].

6. **Study SODA framework** (MLIR dialect for SNN accelerators) and NeuEdge (hardware-aware SNN mapping) to see how others approached the problem [27,28,31].

By the end of this learning path, you'll understand how mainstream AI compilers work (MLIR, Triton, OpenVINO), how Intel's heterogeneous computing strategy (oneAPI/SYCL) fits in, and why neuromorphic hardware remains disconnected from these stacks. You'll be equipped to design the bridge—the missing MLIR-based compiler for neuromorphic hardware.

## Glossary & Acronyms

| Term | Full Form | Definition | Why It Matters |
|------|-----------|------------|----------------|
| AMX | Advanced Matrix Extensions | Intel's matrix multiplication accelerator in Sapphire Rapids and newer CPUs | Enables fast INT8/BF16 matrix operations for AI inference on Intel CPUs |
| AVX-512 | Advanced Vector Extensions 512-bit | Intel's 512-bit SIMD instruction set for vector operations | Critical for high-performance numerical computing on Intel CPUs |
| BYOC | Bring Your Own Codegen | TVM framework for custom accelerator backends | Alternative to IREE for adding hardware support; pattern-matches subgraphs |
| cuBLAS | CUDA Basic Linear Algebra Subroutines | NVIDIA's optimized BLAS library for GPUs | Performance baseline for matrix operations; Triton kernels aim to match it |
| CUDA | Compute Unified Device Architecture | NVIDIA's parallel computing platform and programming model | Dominant GPU programming model; SYCL/Triton aim to provide alternatives |
| DPC++ | Data Parallel C++ | Intel's SYCL-compliant compiler for heterogeneous computing | Implements SYCL standard; enables cross-platform code for Intel (and other) hardware |
| Dialect (MLIR) | - | Namespace containing custom operations, types, attributes in MLIR | Allows domain-specific abstractions (Linalg for ML, GPU for kernels) while sharing infrastructure |
| DSL | Domain-Specific Language | Programming language specialized for a particular domain | Triton is a DSL for GPU kernels; makes kernel programming more accessible |
| FP16 | 16-bit Floating Point | Half-precision floating-point format | Reduces memory bandwidth and enables Tensor Core acceleration on NVIDIA GPUs |
| FP32 | 32-bit Floating Point | Single-precision floating-point format | Standard training precision; target for quantization to INT8/INT4 |
| FP8 | 8-bit Floating Point | Quarter-precision floating-point format | Emerging format for efficient training/inference; supported by H100 GPUs |
| FPGA | Field-Programmable Gate Array | Reconfigurable integrated circuit | Target for custom accelerators; SODA compiles SNNs to FPGAs |
| GCN | Graphics Core Next | AMD's GPU instruction set architecture | Triton compilation target for AMD GPUs |
| HAL (IREE) | Hardware Abstraction Layer | IREE's portable execution interface | Extension point for custom accelerator backends; defines memory/dispatch/sync APIs |
| HLO | High-Level Optimizer | XLA's graph representation for ML computations | Where fusion and scheduling decisions happen in XLA |
| IMEX | Intel Extension for MLIR | Additional MLIR dialects and Python bindings for Intel | Intel's MLIR contributions beyond mainline LLVM |
| INT4 | 4-bit Integer | 4-bit quantized integer format | Aggressive quantization for model compression; 8× smaller than FP32 |
| INT8 | 8-bit Integer | 8-bit quantized integer format | Standard inference quantization; 4× smaller than FP32, 2-4× faster with specialized hardware |
| IREE | Intermediate Representation Execution Environment | Google's MLIR-native compiler and runtime for edge-to-cloud deployment | Portable ML execution across diverse hardware via HAL abstraction |
| IR | Intermediate Representation | Internal compiler format between source and machine code | Enables optimization and multi-target code generation |
| JAX | - | Google's composable transformations library for NumPy | Research framework with functional programming model; compiles via XLA |
| Kernel Fusion | - | Fusing across reduction boundaries and complex control flow | Advanced fusion beyond simple operator chaining (Neptune, 2025) |
| Linalg Dialect | - | MLIR dialect for linear algebra on tensors/buffers | Primary entry point for ML workloads; hardware-agnostic representation of conv, matmul, etc. |
| LLVM | Low Level Virtual Machine | Compiler infrastructure with reusable components | Foundation for modern compilers; MLIR is part of LLVM project |
| Loihi | - | Intel's neuromorphic research chip | Event-driven, asynchronous hardware; lacks MLIR-based compiler |
| Mixed Precision | - | Different numerical precision per layer/operation | Balances accuracy and performance; compiler selects per-layer bit-widths |
| MLIR | Multi-Level Intermediate Representation | LLVM infrastructure for building compilers with extensible dialects and progressive lowering | Foundation for modern AI compilers; enables custom hardware support via dialects |
| MXFP | Microscaling Floating Point | Block-wise floating-point format with shared exponent | Emerging format for efficient ML; MicroMix paper shows 19.7-20% memory reduction [25] |
| NCHW | Number-Channel-Height-Width | Tensor layout with channel dimension before spatial dimensions | Common CNN layout; optimal for some hardware, suboptimal for others |
| NHWC | Number-Height-Width-Channel | Tensor layout with channel dimension after spatial dimensions | Alternative CNN layout; often better for mobile/edge hardware |
| NIR | Neuromorphic Intermediate Representation | Cross-platform interop standard for SNNs | ONNX equivalent for neuromorphic; supports 11 platforms (Nature Comm. 2024) |
| NNCF | Neural Network Compression Framework | OpenVINO's toolkit for PTQ, QAT, pruning | Most mature quantization solution for production deployment |
| NPU | Neural Processing Unit | Specialized AI accelerator | Intel Core Ultra includes NPU; OpenVINO 2025.4 adds NPU support |
| ONNX | Open Neural Network Exchange | Interoperability format for ML models | Enables cross-framework deployment; ONNX-MLIR compiles ONNX via MLIR |
| ONNX-MLIR | - | Compiles ONNX models via MLIR infrastructure | Enables ONNX model deployment through MLIR toolchain |
| oneCCL | oneAPI Collective Communications Library | Communication library for distributed training | Part of oneAPI; handles all-reduce, all-gather for multi-node training |
| oneDAL | oneAPI Data Analytics Library | Data analytics library | Part of oneAPI; accelerates preprocessing and feature engineering |
| oneDNN | oneAPI Deep Neural Network Library | Intel's optimized DNN primitives | Used by PyTorch/TensorFlow for Intel acceleration; part of oneAPI |
| oneMKL | oneAPI Math Kernel Library | Optimized math kernels | Part of oneAPI; BLAS, LAPACK, FFT for CPUs and GPUs |
| oneAPI | - | Intel's unified programming model for heterogeneous computing | Umbrella for DPC++, oneMKL, oneDNN, oneDAL, oneCCL libraries |
| OpenVINO | - | Intel's toolkit for AI inference optimization on Intel hardware | Best-in-class quantization and optimization for Intel CPUs/GPUs/NPUs |
| Operator Fusion | - | Combining operations into single kernels to eliminate memory round-trips | Single most impactful ML compiler optimization; 2-10× speedups common |
| Progressive Lowering | - | Gradual transformation from high-level to hardware-specific code through intermediate representations | Enables optimization at each abstraction level; key to MLIR's flexibility |
| PTQ | Post-Training Quantization | FP32 → INT8/INT4 after training via calibration | Fastest quantization path; 2-4× speedup with <1% accuracy loss (CNNs) |
| PTX | Parallel Thread Execution | NVIDIA's GPU assembly language | Compilation target for CUDA, Triton; intermediate before final GPU binary |
| PyTorch | - | Meta's machine learning framework | Most popular research framework; compiles via Torch-MLIR or torch.compile |
| QAT | Quantization-Aware Training | Training with fake quantization operators | Better accuracy than PTQ for aggressive quantization (INT4, sub-8-bit) |
| ROCm | Radeon Open Compute | AMD's open-source GPU computing platform | Competitor to CUDA; uses LLVM backend, growing MLIR integration |
| SCF | Structured Control Flow | MLIR dialect for loops and conditionals | Represents for/while/if constructs; enables loop optimization passes |
| SIMD | Single Instruction Multiple Data | Parallel execution model applying one operation to multiple data elements | CPU vectorization pattern; Vector dialect targets SIMD units |
| SIMT | Single Instruction Multiple Threads | GPU execution model where many threads execute same instruction on different data | CUDA/GPU programming model; contrasts with SIMD (different hardware) |
| SNN | Spiking Neural Network | Neural network using event-driven spikes instead of continuous activations | Neuromorphic computing model; energy-efficient but lacks mature compiler support |
| SODA Framework | Software Defined Architecture | MLIR dialect for SNN-to-FPGA compilation | First automatic SNN hardware mapping; research prototype [31] |
| SPIR-V | Standard Portable Intermediate Representation | Khronos GPU shader/kernel IR | Target for SYCL/Vulkan/OpenCL compilation; enables vendor-neutral GPU code |
| SSA | Static Single Assignment | IR form where each variable assigned exactly once | Simplifies compiler analysis and optimization; used in MLIR, LLVM |
| StableHLO | - | Stabilized HLO operations for cross-compiler portability | Common interchange format between frameworks and MLIR-based compilers |
| SYCL | - | Khronos standard for heterogeneous C++ computing | Open alternative to CUDA; single-source host/device code |
| TensorFlow | - | Google's machine learning framework | Production-focused framework; compiles via XLA or TFLite |
| Tiling | - | Decomposing large operations into cache-friendly tiles | Essential for memory-bound ML workloads on CPUs and GPUs |
| Torch-MLIR | - | Converts PyTorch models to MLIR dialects | Bridges PyTorch ecosystem to MLIR-based backends |
| TorchAO | - | PyTorch-native model optimization framework | Unifies training and serving quantization; supports FP8, INT4, MXFP formats |
| TPU | Tensor Processing Unit | Google's custom AI accelerator with systolic arrays | Optimized for matrix operations; primary XLA target |
| Transform Dialect | - | MLIR dialect for fine-grained optimization control without recompiling | CGO 2025; enables 10× speedups for DSP workloads [11] |
| Triton | - | OpenAI's Python DSL for GPU kernel programming at block/tile level | Makes GPU kernel development accessible; used by PyTorch, vLLM, DeepSpeed |
| TVM | - | Apache's ML compiler with auto-scheduling and BYOC framework | Alternative to IREE/XLA; strong auto-tuning capabilities |
| USM | Unified Shared Memory | SYCL feature for automatic host/device memory management | Simplifies heterogeneous programming; no manual cudaMemcpy equivalents |
| VNNI | Vector Neural Network Instructions | Intel's INT8 dot-product acceleration (Ice Lake+) | Enables 2-4× speedup for quantized inference on Intel CPUs |
| XeVM Dialect | - | Intel's MLIR dialect for Intel GPU hardware | Upstreamed 2025; shows MLIR extensibility for vendor-specific features [8] |
| XLA | Accelerated Linear Algebra | Google's whole-graph ML compiler for JAX/TensorFlow | Automatic optimization (fusion, layout) for TPU, GPU, CPU without manual tuning |

## How Things Relate (Concept Map)

- **PyTorch/JAX/TensorFlow** → compile via → **Torch-MLIR/StableHLO/ONNX-MLIR** → common target → **MLIR Linalg Dialect**
- **MLIR Linalg** → lowers through → **SCF/Affine** → hardware-specific → **GPU/Vector/LLVM Dialects** → generates → **Machine Code**
- **MLIR** → foundation for → **Triton, IREE, XLA, Intel XeVM, CUDA Tile IR** (extensible ecosystem)
- **Triton** → compiles → **Triton-IR** → lowers → **LLVM IR** → targets → **PTX (NVIDIA), GCN (AMD), SPIR-V (Intel)**
- **XLA** → uses → **HLO graph** → optimizes via → **fusion, layout, buffer assignment** → targets → **TPU, GPU, CPU**
- **IREE** → uses → **HAL** → enables → **Custom Backend** (via PluginAPI) → could target → **Loihi/neuromorphic**
- **OpenVINO** → includes → **NNCF** → provides → **PTQ, QAT, pruning, mixed precision**
- **oneAPI** → includes → **DPC++ (compiler), oneDNN (primitives), oneMKL (math), oneCCL (communication)**
- **SYCL standard** → implemented by → **DPC++** → compiles to → **SPIR-V** → runs on → **Intel/AMD/NVIDIA via backends**
- **Operator Fusion** → eliminates → **memory round-trips** → improves → **inference performance** (2-10× speedups)
- **Quantization (PTQ/QAT)** → reduces → **model size, latency** → enables → **edge deployment, faster inference**
- **Neuromorphic hardware (Loihi)** → programmed via → **Lava** → lacks → **MLIR-based compiler** (research gap)
- **NIR** → provides → **cross-platform SNN interop** → analogous to → **ONNX for traditional DNNs**
- **SODA Framework** → extends → **MLIR** → targets → **FPGA neuromorphic accelerators** (research prototype)
- **torch.compile** → uses → **TorchInductor** → generates kernels via → **Triton** → runs on → **NVIDIA/AMD GPUs**

The key insight: MLIR is the connective tissue. It enables different compilers (Triton, XLA, IREE, OpenVINO, Intel XeVM) to share infrastructure while targeting different hardware. The neuromorphic gap: Loihi/Lava sits outside this ecosystem, lacking the MLIR integration that would enable mainstream framework compilation to neuromorphic targets.

## References

[1] LLVM/MLIR, "MLIR: Multi-Level Intermediate Representation," 2025. [Online]. Available: https://mlir.llvm.org/

[2] OpenAI, "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations," triton-lang/triton GitHub, 2025. [Online]. Available: https://github.com/triton-lang/triton

[3] Google, "IREE: Intermediate Representation Execution Environment," iree-org/iree GitHub, 2025. [Online]. Available: https://github.com/iree-org/iree

[4] Intel, "oneAPI DPC++ Compiler," intel/llvm GitHub, sycl branch, 2025. [Online]. Available: https://github.com/intel/llvm

[5] Intel, "OpenVINO Toolkit 2025.4 Release Notes," Intel Developer Zone, 2025. [Online]. Available: https://www.intel.com/content/www/us/en/developer/articles/release-notes/openvino/2025-4.html

[6] Intel, "Intel Neuromorphic Computing Research," Intel Research, 2025. [Online]. Available: https://www.intel.com/content/www/us/en/research/neuromorphic-computing.html

[7] Intel, "Intel Advances Neuromorphic with Loihi 2, New Lava Software Framework," Intel Newsroom, 2021. [Online]. Available: https://www.intc.com/news-events/press-releases/detail/1502/

[8] Phoronix, "Intel Upstreams XeVM MLIR Dialect Into LLVM," 2025. [Online]. Available: https://www.phoronix.com/news/Intel-XeVM-MLIR-In-LLVM

[9] Modular, "What about the MLIR compiler infrastructure?" 2025. [Online]. Available: https://www.modular.com/blog/democratizing-ai-compute-part-8-what-about-the-mlir-compiler-infrastructure

[10] Built In, "The Next Wave of AI Infrastructure Must Target NVIDIA's CUDA Moat," 2025. [Online]. Available: https://builtin.com/articles/nvidias-cuda-future-ai-infrastructure

[11] M. Steuwer et al., "The MLIR Transform Dialect," arXiv:2409.03864, 2024. [Online]. Available: https://arxiv.org/abs/2409.03864

[12] "DSP-MLIR: A MLIR Dialect for Digital Signal Processing," arXiv:2408.11205, 2024. [Online]. Available: https://arxiv.org/abs/2408.11205

[13] "Towards High-Performance AI Compiler with Upstream MLIR," arXiv:2404.15204, 2024. [Online]. Available: https://arxiv.org/pdf/2404.15204

[14] "DNNFusion: Accelerating Deep Neural Networks with Advanced Operator Fusion," arXiv:2108.13342. [Online]. Available: https://arxiv.org/abs/2108.13342

[15] "oneDNN Graph Compiler: Hybrid Approach for DL Compilation," arXiv:2301.01333, 2024. [Online]. Available: https://arxiv.org/abs/2301.01333

[16] "Applying Graph Explanation to Operator Fusion," arXiv:2501.00636, 2024. [Online]. Available: https://arxiv.org/abs/2501.00636

[17] "ML-Triton: Multi-Level Compilation Extension to Triton," arXiv:2503.14985, 2025. [Online]. Available: https://arxiv.org/abs/2503.14985

[18] "Iris: First-Class Multi-GPU Programming in Triton," arXiv:2511.12500, 2025. [Online]. Available: https://arxiv.org/html/2511.12500

[19] "Linear Layouts: Robust Code Generation Using F2," arXiv:2505.23819, 2025. [Online]. Available: https://arxiv.org/html/2505.23819v2

[20] "Neptune: Advanced ML Operator Fusion," arXiv:2510.08726, 2025. [Online]. Available: https://arxiv.org/html/2510.08726v1

[21] "Act: Automatically Generating Compiler Backends," arXiv:2510.09932, 2025. [Online]. Available: https://arxiv.org/html/2510.09932

[22] "QuantuneV2: Compiler-Based Mixed Precision Quantization," arXiv:2501.07161, 2025. [Online]. Available: https://arxiv.org/abs/2501.07161

[23] "Progressive Mixed-Precision Decoding for LLM Inference," arXiv:2410.13461, 2025. [Online]. Available: https://arxiv.org/abs/2410.13461

[24] "TorchAO: PyTorch-Native Model Optimization," arXiv:2507.16099, 2025. [Online]. Available: https://arxiv.org/html/2507.16099v1

[25] "MicroMix: Microscaling Formats for LLMs," arXiv:2508.02343, 2025. [Online]. Available: https://arxiv.org/html/2508.02343v1

[26] "Spiker+: Efficient SNN FPGA Accelerators," arXiv:2401.01141, 2023. [Online]. Available: https://arxiv.org/html/2401.01141v1

[27] "NeuEdge: Energy-Efficient Neuromorphic Computing for Edge AI," arXiv:2602.02439, 2025. [Online]. Available: https://arxiv.org/html/2602.02439

[28] "Compiling Spiking Neural Networks to Neuromorphic Hardware," arXiv:2004.03717, 2020. [Online]. Available: https://arxiv.org/abs/2004.03717

[29] "SNNAX: Spiking Neural Networks in JAX," arXiv:2409.02842, 2024. [Online]. Available: https://arxiv.org/html/2409.02842v1

[30] "Neuromorphic Intermediate Representation (NIR)," *Nature Communications*, 2024. [Online]. Available: https://www.nature.com/articles/s41467-024-52259-9

[31] "SODA Framework: MLIR Dialect for SNN Accelerator Generation," in *Proc. IEEE/ACM ICCAD*, 2022. [Online]. Available: https://dl.acm.org/doi/10.1145/3508352.3549424
