# CUDA Histogram Construction — Optimization Journey

GPU used: **NVIDIA A30** (24 GB HBM2, driver 580.126.20, CUDA 13.0)

This document tracks each optimization stage of a CUDA histogram kernel, from a naive globally-contended implementation up to a warp-aggregated version using warp-level intrinsics.

---

## Problem setup

The input is a large array of characters representing an image's pixel values, each in the range **0–255** (256 possible values). For this exercise the array holds **64 million elements** (64 × 2²⁰ ≈ 64 × 10⁶).

The goal is to build a **frequency histogram**: x-axis = pixel value (0–255), y-axis = how many times that value occurs across the whole input array. This is the basis for real-world operations like brightness/contrast adjustment in cameras, photo editors, and car image-recognition systems, where a device computes a histogram once and then applies a calculated edit across the entire image.

---

## Version 1 — Naive Histogram (global contention)

The baseline approach: every thread reads one element of the input array using a memory-coalesced access pattern (thread 0 reads element 0, thread 1 reads element 1, etc.), then does an **atomic increment** directly on the global histogram.

**Grid-stride loop:** since consumer/datacenter GPUs only have a limited number of SMs, we pick a grid size that lets the whole grid cover a stretch of the input per iteration, then loop over the remaining array in a grid-stride fashion — each stride covering `gridDim * blockDim` elements. The exact grid/block dimensions are chosen based on the underlying hardware's SM count, to size each iteration's memory footprint appropriately.

With **256 blocks × 256 threads per block**, each iteration processes 256 × 256 = 65,536 elements = 64 KB. For a 64 MB input, that's 64 MB / 64 KB = 1,024 iterations total.

**Why this is slow — atomic serialization:**
- For **balanced** data (pixel values spread evenly), contention per histogram bin is roughly: 64 × 10⁶ elements / grid size (256×256) ≈ 4,096 atomics serialized per bin.
- For **skewed** data (most pixels clustered around a few values), in the worst case all 64 × 10⁶ writes could serialize onto the same bin.

This atomic contention is the central bottleneck — every thread hitting the same global memory location for an atomic increment forces hardware-level serialization of those writes.

**Output:**
```
Verification PASSED!
Average time taken by each kernel (i.e. histogram creation is): 58.4243 ms
[naive Histogram bandwidth] is: 1.14866 GB per Sec

-- random (unbalanced) data, to compare against Version 3's warp intrinsics later --
Verification PASSED!
Average time taken by each kernel (i.e. histogram creation is): 61.0642 ms
[naive Histogram bandwidth] is: 1.099 GB per Sec
```

---

## Version 2 — Privatized Histogram (shared memory)

The fix for Version 1's bottleneck: **privatization**. Instead of every thread across the entire grid fighting over one global histogram, each **thread block** builds its own private histogram in shared memory. Once a block finishes, its 256 local bin values are aggregated into the global histogram.

Contention still exists during the final aggregation step — threads at the same bin index across different blocks still contend for the same global location — but the degree of contention drops from "every thread in the grid" down to "at most one write per block per bin," which is far cheaper.

**Grid-stride vs. block-stride — when to use which:**
- If the data lives **once per block** (e.g. `__shared__` memory, a block's own tile) → use a **block-stride loop**.
- If the data lives in **global memory** (the full `d_in`/`d_out` dataset that all blocks cooperate on) → use a **grid-stride loop**.

**Output:**
```
Average time taken by each kernel (i.e. histogram creation is): 0.66815 ms
[Privatized Histogram bandwidth] is: 100.441 GB per Sec
```

Bandwidth improved **~100×**, and kernel time dropped by roughly the same factor — a direct result of moving contention from the full grid down to per-block granularity.

---

## Version 3 — Warp-Aggregated Histogram

A further refinement on top of privatization: within a single **warp** (32 threads), if multiple threads happen to read the same pixel value, they'd normally each fire their own atomic increment — 32 atomics for 32 threads, even though they're all incrementing the same bin.

Instead, using **warp-level shuffle/vote intrinsics**, the warp can first determine how many of its 32 threads share the same value, consolidate that into a single count, and then issue just **one atomic increment** representing the whole warp's contribution to that bin — reducing up to 32 atomics down to 1 in the best case.

**Why this works:** warp voting uses hardware-level intrinsics that let threads within the same warp communicate directly via registers — never touching shared memory — completing the vote in a single clock cycle. This cuts contention dramatically compared to Version 2's per-block atomics.

**Compilation note:** warp-level intrinsics require targeting Volta or newer architectures explicitly:
```bash
nvcc code.cu -arch=sm_80 -O3 -o code.exe   # sm_70 also works (Volta+)
```

**Output:**
```
$ nvcc 03_histogram_warp_aggregated.cu -arch=sm_80 -O3 -o 03_histogram_warp_aggregated
$ ./03_histogram_warp_aggregated
Verification PASSED!
Average time taken by each kernel (i.e. histogram creation is): 0.86483 ms
[Warp aggregated Histogram bandwidth] is: 77.599 GB per Sec
-- lower bandwidth than Version 2 here, because the input data is uniform: each warp
   likely sees close to 32 unique values, so there's little to aggregate, defeating
   the purpose of this optimization --

-- random (unbalanced) data, where warp-aggregation has more duplicate values per warp to collapse --
Verification PASSED!
Average time taken by each kernel (i.e. histogram creation is): 0.825948 ms
[Warp aggregated Histogram bandwidth] is: 81.2519 GB per Sec
```

This highlights an important nuance: warp aggregation only pays off when a warp's 32 threads actually share repeated values. On **uniformly distributed** pixel data, where each thread in a warp tends to see a distinct value, there's nothing to consolidate — so Version 3 can actually underperform Version 2 on that kind of input, while still winning on skewed/repetitive data.

---

## Summary

| Version | Technique | Avg Kernel Time | Bandwidth |
|---|---|---|---|
| 1 | Naive, direct global atomics | 58.42–61.06 ms | ~1.1 GB/s |
| 2 | Privatized per-block histogram in shared memory | 0.668 ms | ~100.4 GB/s |
| 3 | Warp-aggregated atomics (uniform data) | 0.865 ms | ~77.6 GB/s |
| 3 | Warp-aggregated atomics (skewed/random data) | 0.826 ms | ~81.3 GB/s |
