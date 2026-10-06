## Experiment: `e5_1_small_collocation`

### Quantitative Analysis of Scheduling Efficiency

#### Environment and workload

The experiment runs four scheduler variants—Sequential, Preemptive, Spatial, and Virtual—on the same `MeshAcceleratorConfig.small` device. Its processor clock is 1 GHz, so one million simulated cycles equal 1 ms. The device has a `4x4` interconnect, eight compute tiles arranged as a `4x2` CCG mesh, and four DMA tiles. The run uses `--ccg-tops 0.5` by default, specifying 0.5 TOPS per compute tile. Each compute tile uses a `32x32` compute array. The small-device defaults use lightweight interconnect and DRAM timing models (`booksim2_enable=False`, `dramsim3_enable=False`). Compilation uses `bfloat16` tensors and a default `32x32` tile shape.

Every variant uses a scheduler configured with a 10,000,000-cycle starvation threshold and a candidate window of 16. Sequential retains one active workload until it finishes. Preemptive can switch workloads at kernel boundaries but limits dispatch to one running kernel. Spatial enables joint kernel planning and can place kernels concurrently on the shared mesh. Virtual assigns each camera request its own `1x2` domain with two compute tiles and one DMA tile. The four disjoint domains cover all eight compute tiles and all four DMA tiles; requests run concurrently within fixed partitions.

The workload is one simultaneous release of four independent YOLOX-Nano camera-detection requests: `front`, `left`, `right`, and `rear`. All arrive at cycle 0. Each processes one `416x416x3` image with the same YOLOX-Nano configuration (`depth=0.33`, `width=0.25`, 10 classes). Each request carries a 50,000,000-cycle (50 ms) SLO and the same scheduling hint: priority 2, weight 2.0, and maximum wait of 15,000,000 cycles. There are no periodic arrivals, lane-segmentation requests, driver-monitoring requests, or warmup requests. Each run submits and measures exactly four workloads, comprising 156 kernels per request and 624 completed kernels in total.

The analysis uses the current `workload_profile.csv`, `scheduler_profile.csv`, `kernel_profile/kernel_profile_camera.det.csv`, and `metadata.json` in `.cache/run_sequential`, `.cache/run_preemptive`, `.cache/run_spatial`, and `.cache/run_virtual`. Response time is completion minus arrival; queueing is first kernel start minus arrival; execution is completion minus first kernel start. Since all arrivals occur at cycle 0, batch makespan is the last request's completion time. The four responses are shown directly below; percentile estimates from four requests would add little information. All four requests meet the 50 ms SLO under every policy, so deadline misses are only a sanity check here.

| Policy | Mean response (ms) | Batch makespan / worst response (ms) | Mean queueing (ms) | Mean execution (ms) | Deadline misses | Minimum slack (ms) | Simulator wall time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Sequential | 2.358 | 3.772 | 1.415 | 0.943 | 0 / 4 | 46.228 | 56.6 |
| Preemptive | 4.573 | 4.573 | 0.206 | 4.367 | 0 / 4 | 45.427 | 76.0 |
| Spatial | 2.489 | 2.911 | 0.090 | 2.398 | 0 / 4 | 47.089 | 72.3 |
| Virtual | 3.187 | 3.187 | 0.000 | 3.187 | 0 / 4 | 46.813 | 49.4 |

| Camera | Sequential response (ms) | Preemptive response (ms) | Spatial response (ms) | Virtual response (ms) |
|---|---:|---:|---:|---:|
| Front | 0.943 | 4.573 | 2.911 | 3.187 |
| Left | 1.886 | 4.573 | 2.344 | 3.187 |
| Right | 2.829 | 4.573 | 1.994 | 3.187 |
| Rear | 3.772 | 4.573 | 2.705 | 3.187 |

Sequential has the lowest mean response because it completes one camera request in about 0.943 ms before starting the next. Its requests finish in front, left, right, rear order. The last camera therefore waits 2.829 ms before starting, making the batch take 3.772 ms. Virtual starts all four requests at cycle 0 in separate domains. They finish together at 3.187 ms, giving a 15.5% shorter batch makespan than Sequential, although the smaller two-core allocation raises mean response by 35.2%.

Preemptive starts all four cameras within 0.412 ms, reducing mean initial queueing by 85.5% relative to Sequential. Kernel-boundary interleaving then stretches mean first-start-to-completion time from 0.943 to 4.367 ms. All four completions fall within 0.001 ms of 4.573 ms. Its mean response is 93.9% higher and its batch makespan is 21.2% higher than Sequential's. Low initial queueing alone does not improve completion time when the requests repeatedly wait between kernels.

Spatial reduces mean initial queueing to 0.090 ms, 93.6% below Sequential, but its mean execution interval is 2.398 ms because concurrent kernels receive smaller placements. Its mean response is 5.5% above Sequential's. Its batch makespan, however, is 2.911 ms: 22.8% below Sequential and 36.3% below Preemptive. Right finishes first, while front starts at 0.362 ms and finishes last. Spatial also finishes the batch 8.7% sooner than Virtual. The completion order and exact values matter more than an aggregate percentile for this four-request burst.

#### Scheduling and kernel behavior

| Policy | Scheduler decisions | Kernels selected | Multi-kernel decisions | Mean cores per kernel | Core-count stddev | Eight-core kernels |
|---|---:|---:|---:|---:|---:|---:|
| Sequential | 624 | 624 | 0 | 6.205 | 3.057 | 464 / 624 (74.4%) |
| Preemptive | 624 | 624 | 0 | 6.128 | 3.069 | 452 / 624 (72.4%) |
| Spatial | 621 | 624 | 2 | 2.976 | 2.066 | 45 / 624 (7.2%) |
| Virtual | 156 | 624 | 156 | 1.724 | 0.447 | 0 / 624 |

Sequential uses either one core for a MemCopy kernel or all eight cores for other kernels: 160 one-core and 464 eight-core placements. Preemptive has a similar allocation overall, although 12 kernels use two, four, or six cores. Virtual selects four kernels per decision, one from each camera domain. It assigns two cores to 452 convolutions and one core to 172 kernels, averaging 1.724 cores per invocation. The camera domains use disjoint CCG tile pairs: front `0–1`, left `2–3`, right `4–5`, and rear `6–7`. Spatial distributes kernels across one to eight cores, averaging 2.976 cores per invocation. Virtual's four-kernel decisions are concurrent dispatch across separate fixed domains, not joint placement in a shared core pool.

Spatial makes two multi-kernel decisions. At cycle 0, it jointly dispatches the left, right, and rear cameras' initial MemCopy kernels on three distinct single-core placements; the scheduler records a predicted benefit of 66.67%. At cycle 361,500, it jointly dispatches the front camera's initial MemCopy on one core and the right camera's next convolution on four cores, with a predicted benefit of 48.92%. The remaining 619 decisions select one kernel each. These benefit values are planning estimates against predicted sequential service, not measured request-level speedups. Single-kernel decisions can also occur while other kernels are running, so two joint decisions do not imply that overlap occurred only twice.

| Policy | Profiled kernel invocations | Mean kernel latency (us) | Sum of kernel latencies (ms) | Mean summed kernel latency per request (ms) |
|---|---:|---:|---:|---:|
| Sequential | 624 | 6.046 | 3.772 | 0.943 |
| Preemptive | 624 | 7.329 | 4.573 | 1.143 |
| Spatial | 624 | 15.373 | 9.593 | 2.398 |
| Virtual | 624 | 20.430 | 12.748 | 3.187 |

Kernel-profile averages are weighted by invocation count across 66 kernel signatures. The latency sum adds every kernel's start-to-completion interval, including concurrently running kernels. Sequential and Preemptive run one kernel at a time in this trace, so their summed kernel latencies approximately equal batch makespan. Spatial's 9.593 ms of summed kernel time fits into a 2.911 ms batch, showing an average of about 3.30 active kernels over that interval. Virtual's 12.748 ms sum spans a 3.187 ms batch because four isolated domains execute concurrently; the ratio is exactly four active kernels on average. These ratios measure overlap, not the speed of an individual kernel. Under Preemptive, the mean request execution interval exceeds its mean per-request kernel sum by about 3.224 ms, reflecting time spent between that request's kernels while other cameras progress.

The most expensive recurring camera kernels illustrate the placement cost. The first convolution, with input shape `1x208x208x12`, averages 153.069 us under Sequential, 188.069 us under Preemptive, 536.498 us under Spatial, and 373.073 us under Virtual. The input MemCopy, with input shape `1x416x416x3`, averages 136.256, 137.437, 338.591, and 549.071 us in the same order. Spatial's first convolution uses two cores on average, compared with eight under Sequential and Preemptive. Virtual also uses two cores for that convolution, while its four requests execute concurrently in separate domains. These slower kernels increase its aggregate kernel time even as concurrent progress shortens the batch.

The metadata records host-side simulator wall times of 56.6 s for Sequential, 76.0 s for Preemptive, 72.3 s for Spatial, and 49.4 s for Virtual. This timer surrounds `runtime.run()`; it excludes model compilation and CSV export. The three shared-mesh profiles came from a parallel four-variant run, while the corrected Virtual profile came from a subsequent single-variant run. These host times are not directly comparable as scheduling overhead and are not simulated accelerator latency.

The sum of kernel latencies of preemptive scheduler is larger than that of sequential scheduler, since the preemptive scheduler allocates much smaller number of cores to each Conv2d kernel.

### Qualitative Comparison of Scheduler Variants

**Sequential**

One request retains the shared eight-core mesh until completion. This gives the lowest mean response and shortest individual execution, but camera position in the synchronized burst determines response time. It finishes the last request at 3.772 ms.

**Preemptive**

Kernel-boundary switching lets each camera begin early. The four requests then take turns on the shared device, and each remains active for more than 4 ms before finishing. In this batch, that scheduling pattern worsens both mean response and makespan relative to Sequential.

**Spatial**

Joint placement and later concurrent dispatch allow several cameras to progress at once. Per-kernel latency rises because kernels generally receive fewer cores, yet the last camera finishes at 2.911 ms, the best batch makespan. Spatial improves completion of the burst rather than its average response relative to Sequential.

**Virtual**

Each request occupies its own two-core, one-DMA partition, and all four partitions run concurrently. Every camera starts immediately and completes at 3.187 ms. This produces a shorter batch makespan than Sequential but a longer mean response; Spatial completes the batch sooner still. The result reflects equal-sized fixed partitions across the full device, with no dynamic transfer of cores between cameras.

The trace tests one deterministic, simultaneous camera burst with no warmup or later arrivals. It supports a Spatial makespan advantage for this configuration and a Sequential mean-response advantage. Four requests cannot establish tail-latency behavior, sustained throughput, or deadline robustness under repeated contention; the 50 ms SLO is far above every observed response.
