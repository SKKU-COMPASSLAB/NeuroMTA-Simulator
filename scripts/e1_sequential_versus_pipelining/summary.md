# E1. Sequential E/P/D vs Pipelined E/P/D Analysis

## 1. Purpose

This document analyzes **Experiment 1: Sequential E/P/D vs Pipelined E/P/D**.

The purpose of this experiment is to validate whether sequential execution of
Encode, Prefill, and Decode is inefficient for Video Streaming LMM inference,
and whether pipelining improves latency and resource occupancy.

Unlike E0, this experiment performs a new GPU profiling run and emits
fine-grained timeline events:

1. **Encode step**: per-video-chunk vision encoder execution.
2. **Prefill stage**: LLM prefill for each video/text chunk.
3. **Decode stage**: autoregressive generation, measured at decode-token granularity.

The profiling output is then used to construct:

- an observed execution timeline,
- a sequential E/P/D timeline,
- a pipelined E/P/D timeline,
- a CUDA kernel timeline when available.

---

## 2. Measurement Notes

The experiment output consists of:

| File | Description |
|---|---|
| `results.csv` | Per-sample and per-video summary metrics |
| `timelines/*_observed_timeline.csv` | Timeline reconstructed from actual NVTX ranges |
| `timelines/*_sequential_timeline.csv` | Sequential E/P/D replay timeline |
| `timelines/*_pipelined_timeline.csv` | Pipelined E/P/D replay timeline |
| `timelines/*_kernel_timeline.csv` | CUDA kernel start/end timeline from nsys SQLite |

The profiling run records the following NVTX ranges:

| Range | Meaning |
|---|---|
| `ENCODE_STEP` | One video chunk processed by the vision encoder |
| `PREFILL_STAGE` | LLM prefill for one chunk |
| `DECODE_STEP` | One autoregressive decode step |
| `DECODE_STAGE` | Full decode loop for one question |
| `SAMPLE` | One StreamingBench question sample |

The replay preserves the chunk-level dependency:

$$
E_i \rightarrow P_i,
\quad
D \text{ starts after all } P_i
$$

This is important because using a single coarse Prefill span from the first
Prefill event to the last Prefill event incorrectly includes inter-chunk gaps and
Encode intervals as Prefill occupancy. The corrected replay uses the sum of
actual `PREFILL_STAGE` intervals.

---

## 3. Consistency with E0

At first glance, E1 appeared inconsistent with E0 because an earlier replay
showed low Encode pipeline utilization and very high Prefill utilization. The
root cause was not the measured data, but the replay construction.

Measured stage time is consistent across E0 and E1:

| Experiment | Encode | Prefill | Decode | Encode Share | Prefill Share | Decode Share |
|---|---:|---:|---:|---:|---:|---:|
| E0 coarse stage profiling | 203.00 s | 200.54 s | 49.90 s | 44.8% | 44.2% | 11.0% |
| E1 fine-grain event sum | 201.93 s | 180.25 s | 49.61 s | 46.8% | 41.7% | 11.5% |

Therefore, E1 does **not** show that Encode became inherently smaller. Encode
and Prefill remain comparable in measured stage time. The previous low Encode
utilization came from an incorrect Prefill aggregate span in the replay.

Kernel-time ratios also explain part of the E0/E1 difference:

| Experiment / Stage | Encode Kernel / Range | Prefill Kernel / Range | Decode Kernel / Range |
|---|---:|---:|---:|
| E0 | 25.0% | 80.4% | 26.3% |
| E1 | 25.1% | 97.9% | 28.3% |

E1's fine-grained Prefill ranges remove much of the non-kernel gap that E0's
coarser Prefill ranges included. This supports the interpretation that E0
contained extra non-kernel/framework time inside some coarse stage ranges, while
E1 isolates actual stage intervals more tightly.

---

## 4. Workload Summary

| Video | Samples | Encode Steps | Prefill Ranges | Decode Steps | Frames | Prefill Tokens | Max Context Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| `sample_1` | 5 | 128 | 128 | 160 | 256 | 98,187 | 98,315 |
| `sample_2` | 5 | 315 | 315 | 160 | 630 | 241,636 | 241,764 |
| `sample_3` | 5 | 171 | 171 | 160 | 342 | 131,192 | 131,320 |
| `sample_4` | 5 | 220 | 220 | 160 | 440 | 168,813 | 168,941 |
| **Total** | **20** | **834** | **834** | **640** | **1,668** | **639,828** | **241,764** |

---

## 5. Sequential vs Pipelined Latency

## 5.1 Corrected Per-Video Speedup

The corrected pipelined replay reduces the makespan for all four videos.

| Video | Sequential Makespan | Pipelined Makespan | Speedup | Overlap Ratio |
|---|---:|---:|---:|---:|
| `sample_1` | 55.02 s | 31.63 s | 1.74x | 42.5% |
| `sample_2` | 176.71 s | 108.08 s | 1.63x | 38.8% |
| `sample_3` | 84.77 s | 46.93 s | 1.81x | 44.6% |
| `sample_4` | 115.30 s | 65.57 s | 1.76x | 43.1% |

Aggregate result:

| Metric | Value |
|---|---:|
| Average speedup | 1.73x |
| Weighted speedup | 1.71x |
| Speedup range | 1.63x--1.81x |
| Average pipeline resource utilization | 57.8% |
| Average resource waste | 42.2% |

This confirms that sequential execution leaves substantial pipeline parallelism
unused. The corrected speedup is larger than the previous summary because the
previous replay overestimated Prefill occupancy by using a coarse span.

## 5.2 Corrected Resource Occupancy

| Video | Encode Util. | Prefill Util. | Decode Util. | Avg Pipeline Util. | Bottleneck |
|---|---:|---:|---:|---:|---|
| `sample_1` | 91.8% | 45.5% | 36.7% | 58.0% | Encode |
| `sample_2` | 70.7% | 80.8% | 12.0% | 54.5% | Prefill |
| `sample_3` | 90.4% | 63.7% | 26.6% | 60.2% | Encode |
| `sample_4` | 82.5% | 74.2% | 19.2% | 58.6% | Encode |
| **Average** | **83.8%** | **66.0%** | **23.6%** | **57.8%** | - |

The corrected result is now consistent with E0: Encode and Prefill are both
major contributors. The bottleneck changes by video; `sample_2` is Prefill-heavy,
while the other videos are Encode-heavy.

---

## 6. Resource Occupancy Improvement

If the same three logical E/P/D resources are available but execution is purely
sequential, only one stage is active at a time. The average occupancy across the
three resources is therefore approximately:

$$
U_{\text{seq,3-resource}}
= \frac{1}{3}
= 33.3\%
$$

With corrected pipelining, average resource occupancy becomes 57.8%:

$$
\frac{57.8}{33.3} \approx 1.73\times
$$

This matches the corrected average speedup. Pipelining therefore improves both
latency and resource occupancy, but 42.2% of the logical E/P/D capacity still
remains idle because Decode is short and stage balance changes by video.

---

## 7. Compute and Memory Utilization

## 7.1 Kernel-Active Utilization

The resource occupancy numbers in Section 5 measure how long each logical
pipeline is busy over the full pipelined makespan. Compute and memory bandwidth
utilization are different metrics. In this section, utilization is computed
using **CUDA kernel time** as the denominator:

$$
U_{\text{compute,kernel}}
=
\frac{\text{Estimated FLOPs} / T_{\text{kernel}}}{\text{Peak TOPS}}
$$

$$
U_{\text{bandwidth,kernel}}
=
\frac{\text{Estimated Bytes} / T_{\text{kernel}}}{\text{Peak HBM Bandwidth}}
$$

where $T_{\text{kernel}}$ is the sum of CUDA kernel durations overlapping the
corresponding NVTX stage ranges. This removes Python, PyTorch dispatch,
framework gaps, and stage idle time from the denominator.

Using A100-class reference values of 312 TOPS and 1,555 GB/s HBM bandwidth:

| Stage | Kernel / Range Time | Kernel-Active TOPS | Kernel-Active Compute Util. | Kernel-Active Bandwidth | Kernel-Active BW Util. |
|---|---:|---:|---:|---:|---:|
| Encode | 25.1% | 63.95 TOPS | 20.5% | 21.99 GB/s | 1.41% |
| Prefill | 97.9% | 20.94 TOPS | 6.71% | 29.30 GB/s | 1.88% |
| Decode | 27.8% | 1.26 TOPS | 0.40% | 408.53 GB/s | 26.27% |

For comparison, if the same FLOPs and bytes are divided by full NVTX stage
range duration instead of kernel time, the utilization is much lower for Encode
and Decode:

| Stage | Range-Time Compute Util. | Range-Time BW Util. |
|---|---:|---:|
| Encode | 5.14% | 0.35% |
| Prefill | 6.57% | 1.84% |
| Decode | 0.11% | 7.30% |

This distinction is important. Encode looked like a very low-utilization stage
when measured over the full stage range, but during active CUDA kernel execution
it reaches 20.5% of A100-class peak compute. Decode remains compute-light, but
its kernel-active bandwidth utilization reaches 26.3%, showing that Decode is a
short, bursty, memory-bandwidth-sensitive phase.

## 7.2 Stage-Level Event Variability

The timeline files show that stage intensity is not constant. The TOPS and
bandwidth ranges below use kernel time as the denominator.

| Stage | Events | Duration Avg | Duration Range | Kernel TOPS Range | Kernel Bandwidth Range |
|---|---:|---:|---:|---:|---:|
| Encode | 834 | 242.1 ms | 216.9--627.5 ms | 62.72--65.24 | 21.57--22.43 GB/s |
| Prefill | 834 | 216.1 ms | 73.9 ms--1.42 s | 6.76--134.18 | 9.46--184.28 GB/s |
| Decode | 20 | 2.48 s | 2.23--3.02 s | 0.75--1.51 | 293.50--662.28 GB/s |

Fine-grained E1 makes the actual Prefill execution intervals visible instead of
collapsing them into a coarse sample-level span.

---

## 8. Timeline-Based Dynamic Scheduling Analysis

To test whether compute and memory demand changes over time, the corrected
pipelined timelines were divided into 1-second bins. For each bin, active stage
intervals were accumulated and converted into time-weighted TOPS and bandwidth
demand.

## 8.1 Aggregate Time-Binned Variation

Across all four corrected pipelined timelines:

| Metric | Average | Min | P10 | Median | P90 | P99 | Max | CV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TOPS | 27.38 | 0.07 | 8.13 | 33.23 | 36.35 | 38.78 | 39.99 | 0.40 |
| Bandwidth | 46.93 GB/s | 10.14 | 12.30 | 30.24 | 123.61 | 158.81 | 159.73 | 0.90 |
| Active resources | 1.70 | 0.08 | 1.00 | 1.81 | 2.32 | 3.00 | 3.00 | 0.30 |

The result confirms that resource demand is time-varying. Bandwidth remains
especially bursty: the P99 bandwidth demand is 5.25x higher than the median
bandwidth demand.

## 8.2 Per-Video Timeline Variation

| Video | TOPS CV | Bandwidth CV | Active Resource CV | Multi-Stage Bin Share |
|---|---:|---:|---:|---:|
| `sample_1` | 0.32 | 0.64 | 0.30 | 93.8% |
| `sample_2` | 0.48 | 1.08 | 0.33 | 71.6% |
| `sample_3` | 0.28 | 0.74 | 0.26 | 93.6% |
| `sample_4` | 0.36 | 0.87 | 0.29 | 84.8% |

Every video shows non-trivial temporal variation. Therefore, the hypothesis that
compute and bandwidth utilization might remain constant over the pipelined
timeline is not supported by the data.

---

## 9. Root-Cause Analysis of the Previous Inconsistency

The apparent inconsistency was caused by two factors.

1. **Incorrect pipelined timeline aggregation.**  
   The previous replay collapsed all per-chunk Prefill events in a sample into a
   single event whose duration was:

   $$
   T_{P,\text{span}}
   = t_{P,\text{last end}} - t_{P,\text{first start}}
   $$

   This included inter-chunk gaps and Encode intervals. As a result, Prefill
   resource utilization was overestimated and Encode utilization appeared too
   low.

2. **E1 uses finer-grained NVTX ranges than E0.**  
   E0's coarse stage ranges include more non-kernel/framework time. E1 isolates
   per-chunk `ENCODE_STEP` and `PREFILL_STAGE` ranges, so Prefill's kernel/range
   ratio rises from 80.4% in E0 to 97.9% in E1.

After correcting the replay, E1 is consistent with E0: Encode and Prefill are
both large, pipelining improves speedup, and dynamic scheduling is still needed.

---


## 10. Non-Kernel Overhead Interpretation

The low `Kernel / Range Time` for Encode and Decode indicates that a large
fraction of their measured NVTX range duration is not spent inside CUDA kernels.
This is consistent with the code structure.

For **Encode**, `ENCODE_STEP` is emitted by a forward hook around the vision
module. Input construction, video decoding, chat-template processing, and token
metadata extraction happen outside this range. Therefore, the non-kernel portion
inside `ENCODE_STEP` is most likely caused by Python/PyTorch module orchestration
inside the vision encoder, many small kernel launches, CPU-side dispatch gaps,
and framework scheduling overhead between kernels. This matches the workload
shape: video is processed as many small chunks, so each chunk launches a small
sequence of vision kernels rather than one large saturated GPU workload.

For **Decode**, `DECODE_STAGE` wraps the autoregressive token loop, and each
`DECODE_STEP` wraps the decoder-layer stack. The measured Decode kernel/range
ratio is low because every generated token is a small batch-1 operation with a
strict dependency on the previous token. The range also includes loop overhead,
KV-cache object updates, `argmax` token selection, Python-side model dispatch,
and gaps between many small kernels. This supports the interpretation that
Decode is not only memory-sensitive but also launch/dispatch-sensitive at small
batch size.

For **Prefill**, the kernel/range ratio is 97.9%, which means the fine-grained
`PREFILL_STAGE` ranges are mostly occupied by CUDA kernels. Prefill processes
larger token blocks and therefore amortizes Python and kernel launch overhead
much better than Encode and Decode.

Overall, the data supports the proposed explanation: Encode and Decode suffer
from substantial non-kernel overhead because their execution is broken into many
small units, while Prefill is closer to a dense GPU-kernel phase. This is another
reason why software-only pipelining is not enough; the accelerator should reduce
small-stage overheads and schedule fine-grained Encode/Decode work more
explicitly.

---

## 11. Conclusion

Experiment 1 validates the benefit and limitation of E/P/D pipelining.

1. **Pipelining improves latency.**  
   Corrected pipelined E/P/D replay reduces makespan by 1.63x--1.81x, with an
   average speedup of 1.73x and a weighted speedup of 1.71x.

2. **Pipelining improves resource occupancy.**  
   Average three-resource occupancy increases from the sequential baseline of
   33.3% to 57.8%.

3. **The corrected result is consistent with E0.**  
   E1 measured stage shares are 46.8% Encode, 41.7% Prefill, and 11.5% Decode,
   close to E0's 44.8%, 44.2%, and 11.0%.

4. **Resource imbalance remains.**  
   Decode resource occupancy averages only 23.6% because the output length is
   short, and the bottleneck alternates between Encode and Prefill depending on
   video. Kernel-active Decode bandwidth utilization still reaches 26.3%, so
   Decode should be treated as a bursty memory-sensitive phase.

5. **Dynamic scheduling is necessary.**  
   The corrected timeline still shows time-varying compute and memory demand.
   Bandwidth CV is 0.90, and P99 bandwidth is 158.81 GB/s versus a median of
   30.24 GB/s. This supports dynamic CCG allocation and TPOT-aware HBM
   scheduling in the proposed EPD-specialized accelerator.
