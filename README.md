# HPC101-MoE-Optimization

Mixture of Experts (MoE) parallel optimization using OpenMP on ZJU HPC cluster.

## Task
Optimize MoE (Mixture of Experts) dispatch and computation kernel to achieve maximum speedup over the reference implementation across 4 scenarios with varying matrix dimensions and expert counts.

## Results

| Scenario | Speedup | 100pt Threshold | 120pt Threshold |
|----------|---------|-----------------|-----------------|
| S1 | 38.12x | 20x | 26x |
| S2 | 43.75x | 15x | 19x |
| S3 | 151.32x | 100x | 150x |
| S4 | 950.81x | 120x | 275x |

All scenarios pass the 100pt line. S1, S2, S4 pass the 120pt line.

## Key Optimization
- `student/moe_opt.cpp`: Parallel MoE dispatch with OpenMP, cache-friendly memory access patterns, loop unrolling, and SIMD vectorization

## Build
```bash
cmake -B build && cmake --build build -j
```

## Run
```bash
./build/lab2 <scenario> <M> <N> <K> <experts> <iterations>
```
