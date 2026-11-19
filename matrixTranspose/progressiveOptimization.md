# CUDA Matrix Transpose — Optimization Journey

GPU used: **NVIDIA A30** (24 GB HBM2, driver 580.126.20, CUDA 13.0)

This document tracks each optimization stage of a CUDA matrix-transpose kernel, from a naive baseline up to a bank-conflict-free shared-memory implementation, measured against the hardware's Speed of Light (SOL) memory bandwidth.

---

## Version 1 — Plain Matrix Copy (measuring Speed of Light)

Transposing a matrix involves **zero FLOPs** — it's a pure memory-traffic operation. So before optimizing the transpose itself, we need a baseline: the bare-metal memory bandwidth this GPU can sustain on a simple copy, with no transpose-induced access pattern at all.

**Memory path for this version:**
- **Source (read):** matrix `d_in`, already resident in the GPU's off-chip VRAM (HBM/GDDR)
- **Through the chip:** data travels across the GPU's high-speed memory bus, through L2 cache, then L1 cache, landing briefly in registers inside the Streaming Multiprocessors (SMs)
- **Destination (write):** matrix `d_out`, a separate buffer also allocated in VRAM

Both the read and the write here are fully **coalesced** — N² reads, N² writes, no awkward access pattern.

```
Avg Time: 0.195052 ms | Effective Bandwidth: 688.114 GB/s  (N = 4096)
Avg Time: 0.768297 ms | Effective Bandwidth: 698.780 GB/s  (N = 8192)
```

This ~690–700 GB/s is our **SOL baseline** — everything below is measured against it.

---

## Version 2 — Naive Transpose

Here we actually transpose. The **reads** are coalesced (same as Version 1), but by definition, the output matrix's layout relative to the input forces the **writes** to be vertical — i.e., uncoalesced.

**Why uncoalesced writes are so expensive**, for a warp of 32 threads:

- **Coalesced access (read):** 32 threads access nearby memory locations → the transaction pulls in 4 sectors × 32 bytes = 128 bytes total, and all 128 bytes are used. **Bandwidth efficiency: 100%.**
- **Uncoalesced access (write):** each of the 32 threads needs a different cache line → 32 × 128 bytes gets fetched, of which only 128 bytes are actually written. Bandwidth utilized = 128 / (128 × 32) = **3.25%** in theory.

In practice, modern GPU architectures optimize this up from 100/32 (3.25%) to **100/8 (12.5%)** — instead of pulling a whole 128-byte cache line per write, each write only pulls a single 32-byte sector.

This is exactly what we observe: bandwidth drops from the ~688 GB/s SOL baseline to **~86 GB/s**, which is almost exactly 12.5% of baseline.

```
[Version 2: naive transpose] Avg Time: 1.54526 ms | Effective Bandwidth: 86.8579 GB/s  (N = 4096)
[Version 2: naive transpose] Avg Time: 5.20554 ms | Effective Bandwidth: 103.134 GB/s (N = 8192)
```

---

## Version 3 — Shared Memory Transpose (bank conflicts present)

To get coalesced reads **and** coalesced writes on global memory, we need a staging buffer in between — **shared memory**. Each block loads a `TILE × TILE` chunk of the input into shared memory, then writes it back out to global memory in a transposed but still-coalesced fashion.

**Mental model of the flow:**

1. All blocks that have SM resources available launch at the same time.
2. Each block has 256 threads but needs to cover a `32 × 32` (1024-cell) tile, so each thread handles 4 cells (256 × 4 = 1024).
3. The first 256 cells (the tile's top 8 rows) get copied into shared memory simultaneously by all 256 threads.
4. A loop then iterates vertically to fill the remaining 3 chunks of the tile.
5. To write the output in a coalesced manner, chunks are read back out of shared memory left-to-right, and written into the output's corresponding tile top-to-bottom.

**The catch:** reading shared memory *vertically* here causes a **32-way bank conflict** in every tile, across every block — shared memory has exactly 32 banks, and a vertical read pattern on a `32`-wide tile means every thread in the warp lands on the same bank.

Even so, the penalty from a shared-memory bank conflict is much smaller than the penalty from an uncoalesced global-memory write, so bandwidth improves substantially over Version 2:

```
[Version 3: shared memory transpose] Avg Time: 0.401612 ms | Effective Bandwidth: 334.198 GB/s (N = 4096)
[Version 3: shared memory transpose] Avg Time: 1.42568 ms  | Effective Bandwidth: 376.571 GB/s (N = 8192)
```

---

## Version 4 — Padded Shared Memory Transpose (no conflict)

Nearly identical to Version 3, with one change: the shared-memory tile is declared with **33 columns instead of 32** (`tile[32][33]`), while only the `32×32` region is actually used for data. The extra column exists purely to eliminate bank conflicts.

**Why padding works:** shared memory has 32 banks, and bank index is (roughly) `address % 32`. In Version 3, the tile is laid out as `[32][32]`, so a fixed column index always maps to the *same* bank across every row — hence the 32-way conflict on a vertical read. By widening each row to 33 elements, every step down a column shifts the starting bank by one (since the row stride is no longer a multiple of 32). This means a warp reading down a column (`tile[threadIdx.x][threadIdx.y + j]`) now hits **32 distinct banks simultaneously**, in a single clock cycle — no conflict at all.

This pushes performance right up near — and in this case slightly *above* — the SOL baseline, which is explained by L2 cache effects and burst efficiency on this hardware:

```
[Version 4: padded shared memory transpose] Avg Time: 0.175879 ms | Effective Bandwidth: 763.127 GB/s (N = 4096)
[Version 4: padded shared memory transpose] Avg Time: 0.704383 ms | Effective Bandwidth: 762.186 GB/s (N = 8192)
```

---

## Summary

| Version | Technique | Bandwidth (N=4096) | % of SOL |
|---|---|---|---|
| 1 | Plain copy (SOL baseline) | 688.1 GB/s | 100% |
| 2 | Naive transpose (uncoalesced writes) | 86.9 GB/s | ~12.6% |
| 3 | Shared memory (32-way bank conflict) | 334.2 GB/s | ~48.6% |
| 4 | Padded shared memory (no bank conflict) | 763.1 GB/s | ~110.9% |
