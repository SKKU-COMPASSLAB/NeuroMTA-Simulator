# E0. Isolated E/P/D Stage Profiling Analysis

## 1. Purpose

This document analyzes the first experiment for validating the motivation of the
**On-Device EPD-Specialized Accelerator for Video Streaming LMMs**.

The experiment profiles four StreamingBench videos with the
`Qwen/Qwen2.5-Omni-3B` model and decomposes each sample into three isolated
stages:

1. **Encode**: Vision encoder processing sampled video frames.
2. **Prefill**: LLM processing of visual tokens and text prompt tokens.
3. **Decode**: Autoregressive generation of 32 output tokens.

The goal is to determine whether the measured stage behavior supports the need
for a heterogeneous EPD accelerator with:

- CCG-Encode for video encoding
- CCG-Prefill for LLM prefill
- DCG-Decode for memory-bandwidth-sensitive token generation
- HBM shared by Prefill and Decode
- LPDDR for vision weights and cold visual KV cache

---

## 2. Measurement Notes

The profiling output is stored in `results.csv`. Each row corresponds to one
question sample from one video. There are 20 rows in total:

- 4 videos
- 5 question samples per video
- 32 generated decode tokens per sample

The latency columns are measured from NVTX ranges:

| Metric | Meaning |
|---|---|
| `sample_wall_ms` | End-to-end wall time of the sample range |
| `vision_encode_ms` | Total duration of vision encode ranges |
| `prefill_ms` | Total duration of LLM prefill ranges |
| `decode_ms` | Total duration of decode-loop ranges |
| `*_kernel_ms` | CUDA kernel time overlapping each stage |

The TOPS and bandwidth columns are theoretical estimates derived from model
configuration, token count, dtype size, and measured stage duration:

$$
\text{TOPS}_i =
\frac{\text{Estimated FLOPs}_i}{T_i}
$$

$$
\text{Bandwidth}_i =
\frac{\text{Estimated Bytes}_i}{T_i}
$$

Therefore, the bandwidth values should be interpreted as **roofline-style
theoretical bandwidth demand**, not as directly measured HBM traffic. This is
especially important because the measured CUDA memcpy bytes only capture
explicit memcpy activity and do not include all implicit kernel memory traffic.

---

## 3. Workload Summary

The profiled samples cover a wide range of video-context sizes.

| Metric | Average | Min | Median | Max |
|---|---:|---:|---:|---:|
| Sampled video frames | 83.4 | 2 | 90 | 208 |
| Merged visual tokens | 30,024 | 720 | 32,400 | 74,880 |
| Prefill tokens | 31,991 | 806 | 34,511 | 79,822 |
| Context tokens before Decode | 71,379 | 3,129 | 57,201 | 241,764 |
| Decode tokens | 32 | 32 | 32 | 32 |
| Sample wall time | 30.10 s | 2.76 s | 20.74 s | 121.86 s |

The workload is strongly video-context-heavy. Prefill token count scales with
the sampled visual token count, while Decode context length continuously grows
as video context accumulates.

---

## 4. Stage Latency Breakdown

## 4.1 Aggregate E/P/D Breakdown

Across all 20 samples, the sum of E/P/D stage time is 453.43 seconds.

| Stage | Total Time | Share of E/P/D Time | Average Per Sample |
|---|---:|---:|---:|
| Encode | 203.00 s | 44.8% | 10.15 s |
| Prefill | 200.54 s | 44.2% | 10.03 s |
| Decode | 49.90 s | 11.0% | 2.49 s |

Encode and Prefill dominate total execution time, each contributing about
45% of isolated E/P/D time. Decode contributes only about 11% of stage time
because each sample generates a fixed and relatively short 32-token response.

However, Decode is not negligible from an SLO perspective. The average Decode
latency is 2.49 seconds for 32 tokens, corresponding to:

$$
\text{TPOT}_{avg}
=
\frac{2.49 \text{ s}}{32}
\approx
78.0 \text{ ms/token}
$$

The observed TPOT range is 71.0--89.3 ms/token. For interactive streaming
assistants, this makes Decode latency directly visible to the user even though
Decode is not the largest contributor to total E/P/D time.

## 4.2 Per-Video Breakdown

| Video | Wall Time | E/P/D Time | E/P/D Share of Wall | Encode | Prefill | Decode |
|---|---:|---:|---:|---:|---:|---:|
| `sample_1` | 75.47 s | 70.38 s | 93.3% | 44.6% | 37.9% | 17.5% |
| `sample_2` | 264.21 s | 176.41 s | 66.8% | 43.4% | 49.4% | 7.2% |
| `sample_3` | 108.55 s | 90.39 s | 83.3% | 46.1% | 40.2% | 13.7% |
| `sample_4` | 153.74 s | 116.25 s | 75.6% | 45.9% | 43.3% | 10.8% |

The per-video breakdown shows two important patterns.

First, Encode and Prefill are both large, but their relative dominance changes
with the sample. For example, `sample_2` becomes Prefill-heavy as accumulated
context grows, while other videos are closer to balanced Encode/Prefill
execution.

Second, `sample_wall_ms` is larger than the sum of isolated E/P/D ranges. The
unattributed portion is 148.54 seconds in total, or 24.7% of wall time. This
likely includes data preparation, synchronization, framework overhead, CPU-side
work, and gaps between profiled ranges. The accelerator-level conclusion should
therefore focus on the isolated E/P/D ranges, while end-to-end system work
should later include these non-kernel overheads.

## 4.3 Pipeline Opportunity

If the three stages are executed sequentially, the stage latency is:

$$
T_{\text{seq}} = T_E + T_P + T_D
$$

If E/P/D are pipelined across specialized resources, the ideal steady-state
stage interval becomes:

$$
T_{\text{pipe}} \approx \max(T_E, T_P, T_D)
$$

Using the measured E/P/D durations:

| Metric | Ideal E/P/D Pipeline Speedup |
|---|---:|
| Average across samples | 2.10x |
| Median across samples | 2.09x |
| Min | 1.66x |
| Max | 2.61x |
| Weighted by total stage time | 1.92x |

This does not include synchronization, NoC transfer, DMA, or memory arbitration
overhead. Still, the result supports the core EPD hypothesis: sequential
monolithic execution leaves substantial pipeline-level parallelism unused.

---

## 5. TOPS Utilization Analysis

For interpretation, the following utilization percentages compare the estimated
TOPS against an A100-class FP16 peak of 312 TFLOP/s.

| Stage | Average TOPS | Min | Max | Avg Compute Utilization |
|---|---:|---:|---:|---:|
| Encode | 44.04 | 13.85 | 88.18 | 14.1% |
| Prefill | 53.21 | 1.92 | 150.07 | 17.1% |
| Decode | 0.34 | 0.10 | 0.87 | 0.11% |

Weighted by total stage time and estimated FLOPs:

| Stage | Weighted TOPS | Weighted Compute Utilization |
|---|---:|---:|
| Encode | 63.18 | 20.2% |
| Prefill | 67.23 | 21.5% |
| Decode | 0.35 | 0.11% |

Encode and Prefill are the only stages with meaningful compute demand. Their
TOPS are still far below A100 peak, which is expected for real framework
execution with kernel launch overheads, imperfect tensor shapes, attention
overheads, and stage boundaries. Nevertheless, their arithmetic intensity is
extremely high:

| Stage | Aggregate Arithmetic Intensity |
|---|---:|
| Encode | 479,841 FLOP/byte |
| Prefill | 91,707 FLOP/byte |
| Decode | 3.09 FLOP/byte |

This sharp contrast is the key architectural signal. Encode and Prefill should
be mapped to a compute-oriented fabric, while Decode should not be evaluated by
peak FLOPS alone.

---

## 6. Memory Bandwidth Utilization Analysis

For interpretation, the following utilization percentages compare estimated
bandwidth demand against an A100-class HBM bandwidth of 1,555 GB/s.

| Stage | Average Bandwidth | Min | Max | Avg Bandwidth Utilization |
|---|---:|---:|---:|---:|
| Encode | 0.64 GB/s | 0.05 GB/s | 5.45 GB/s | 0.04% |
| Prefill | 1.33 GB/s | 0.21 GB/s | 2.55 GB/s | 0.09% |
| Decode | 111.71 GB/s | 85.03 GB/s | 168.97 GB/s | 7.18% |

Weighted by total stage time and estimated bytes:

| Stage | Weighted Bandwidth | Weighted Bandwidth Utilization |
|---|---:|---:|
| Encode | 0.13 GB/s | 0.01% |
| Prefill | 0.73 GB/s | 0.05% |
| Decode | 112.91 GB/s | 7.26% |

Decode has much lower TOPS than Encode and Prefill, but it requires roughly:

- 173x higher average bandwidth than Encode
- 84x higher average bandwidth than Prefill

The estimated Decode traffic is 6.29--15.08 GB per generated token depending on
context length, with an average of 8.80 GB/token. This directly explains why
Decode is memory-sensitive:

$$
B_{\text{decode/token}}
\approx
B_{\text{LLM weights}}
+
B_{\text{KV cache}}(L_{\text{ctx}})
$$

As context grows, the KV component increases. In the measured data, context
length is almost perfectly correlated with Decode bandwidth:

| Relationship | Correlation |
|---|---:|
| Context tokens vs. Decode latency | 0.951 |
| Context tokens vs. Decode bandwidth | 0.996 |
| Visual tokens vs. Decode bandwidth | 0.878 |

This confirms that streaming video context directly increases Decode memory
pressure through accumulated KV cache.

---

## 7. Main Observations

## 7.1 Encode and Prefill Need Compute-Oriented Resources

Encode latency is almost perfectly correlated with sampled frame count and
visual token count:

| Relationship | Correlation |
|---|---:|
| Sampled frames vs. Encode latency | 0.9998 |
| Visual tokens vs. Encode latency | 0.9998 |
| Prefill tokens vs. Encode latency | 0.9999 |

Prefill latency is also strongly correlated with context size:

| Relationship | Correlation |
|---|---:|
| Prefill tokens vs. Prefill latency | 0.860 |
| Context tokens vs. Prefill latency | 0.948 |

This means the CCG cannot use a fixed Encode/Prefill split for all workload
phases. When many frames are sampled, CCG-Encode needs more compute tiles. When
visual context and prompt tokens grow, CCG-Prefill needs more compute tiles.

## 7.2 Decode Needs Bandwidth, Not Peak FLOPS

Decode uses only 0.34 average TOPS, but it consumes 111.71 GB/s average
estimated bandwidth. Its arithmetic intensity is only 3.09 FLOP/byte, which is
orders of magnitude lower than Encode and Prefill.

This supports a dedicated **Decode-CoreGroup (DCG)** optimized for:

- LLM weight streaming
- KV cache read bandwidth
- Attention score computation over long context
- On-the-fly dequantization
- Predictable TPOT
- HBM bandwidth reservation

Peak compute alone is not the right design target for Decode.

## 7.3 Prefill and Decode Should Share HBM, but Need Arbitration

Prefill and Decode use the same LLM weights and KV cache. Splitting them across
separate physical devices would create weight duplication or KV migration
overhead.

The proposed on-device architecture avoids this by placing:

- CCG-Prefill near HBM
- DCG-Decode near HBM
- LLM weights once in shared HBM
- Prefill-generated KV directly consumable by Decode

However, shared HBM also creates a scheduling problem. Even if Prefill is
compute-oriented, it still reads LLM weights and writes KV cache. Decode has a
much higher bandwidth sensitivity and therefore needs TPOT-aware HBM
arbitration.

The measured Decode bandwidth range of 85--169 GB/s implies that the DCG should
receive a guaranteed HBM bandwidth budget during active generation. Otherwise,
Prefill bursts can increase TPOT variance.

## 7.4 Video Streaming Intensifies KV Pressure

The maximum measured context length is 241,764 tokens. Even with GQA, this
creates a large KV working set. The experiment shows that as video context
accumulates:

- Decode estimated bytes per token increases.
- Decode bandwidth rises from 85 GB/s to 169 GB/s.
- Decode latency rises from 2.27 s to 2.86 s for 32 tokens.

This supports the HBM/LPDDR hierarchy in the proposed architecture:

| Data | Preferred Placement |
|---|---|
| LLM weights | HBM |
| Active text KV | HBM |
| Recent visual KV | HBM |
| Query-relevant visual KV | HBM |
| Vision encoder weights | LPDDR |
| Cold visual KV | LPDDR |
| Old video segments | LPDDR or compressed LPDDR |

Decode should not scan LPDDR every token. Cold visual KV should be promoted to
HBM only when it becomes relevant to the current query.

---

## 8. Architectural Implications

## 8.1 CCG-Encode / CCG-Prefill Partitioning

The aggregate E/P/D latency shares are:

| Stage | Share |
|---|---:|
| Encode | 44.8% |
| Prefill | 44.2% |
| Decode | 11.0% |

A first-order balanced pipeline should therefore allocate comparable compute
capacity to Encode and Prefill. However, per-sample behavior varies:

- Encode share ranges from 4.8% to 59.7%.
- Prefill share ranges from 31.0% to 60.2%.
- Decode share ranges from 4.0% to 47.2%.

This variation supports a reconfigurable CCG partition rather than a static
left/right split. The architecture should allow CCG-Encode and CCG-Prefill tile
counts to change according to sampled frame rate, visual token count, prompt
length, and accumulated context.

## 8.2 DCG Bandwidth Reservation

Decode has low compute utilization but high bandwidth demand. Therefore, DCG
should be designed around effective memory bandwidth, not around large GEMM
throughput.

The HBM scheduler should support:

- Minimum bandwidth reservation for DCG
- TPOT-aware priority during active Decode
- Controlled Prefill bandwidth during Decode overlap
- KV write scheduling that does not block Decode KV reads
- Burst absorption for long-context visual queries

This directly corresponds to the experiment schedule's RQ2 and RQ4.

## 8.3 Shared HBM for Prefill and Decode

The result supports on-device E/P/D specialization rather than cloud-style
physical separation. Prefill and Decode should be specialized but not placed in
fully independent memory domains, because:

- Prefill writes the KV cache that Decode immediately consumes.
- Both stages require LLM weights.
- KV migration would be expensive for long video contexts.
- Weight duplication would waste precious on-device memory capacity.

The proposed CCG-Prefill + DCG + shared HBM structure matches this requirement.

## 8.4 LPDDR as Capacity Relief, Not Decode Working Memory

Encode has very low estimated bandwidth demand compared with Decode. This
supports placing vision encoder weights in LPDDR and reserving HBM for LLM
weights and active KV.

LPDDR should primarily serve:

- Vision encoder weight storage
- Cold visual KV storage
- Compressed historical video context
- Retrieval metadata and summaries

HBM should remain the working set for Decode. The measured context sensitivity
shows that repeatedly accessing cold visual KV from LPDDR during every Decode
token would likely harm TPOT.

---

## 9. Conclusion

The first isolated profiling experiment supports the main motivation of the
EPD-specialized accelerator.

1. **Sequential E/P/D execution exposes pipeline parallelism.**  
   The ideal isolated-stage pipeline speedup is 1.92x weighted by total stage
   time and 2.10x on average across samples.

2. **Encode and Prefill dominate latency and are compute-oriented.**  
   They account for 89.0% of total E/P/D time and show much higher TOPS than
   Decode.

3. **Decode is memory-bandwidth-sensitive.**  
   Decode uses only 0.11% of A100-class FP16 peak compute on average, but
   consumes about 112 GB/s estimated bandwidth and has only 3.09 FLOP/byte
   arithmetic intensity.

4. **Streaming video context increases Decode pressure.**  
   Context length is strongly correlated with Decode latency and almost
   perfectly correlated with Decode bandwidth.

5. **The proposed architecture matches the measured asymmetry.**  
   CCG should handle compute-heavy Encode and Prefill, DCG should protect
   bandwidth-sensitive Decode, HBM should be shared by Prefill and Decode, and
   LPDDR should hold vision weights and cold visual KV.

Overall, the result validates the need for heterogeneous E/P/D specialization
and motivates the next experiments on concurrent Prefill/Decode interference,
dynamic CCG partitioning, and TPOT-aware HBM arbitration.
