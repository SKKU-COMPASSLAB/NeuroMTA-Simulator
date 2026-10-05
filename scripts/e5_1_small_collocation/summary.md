## Experiment: `e5_1_small_collocation`

### Quantitative Analysis of Scheduling Efficiency

The result set contains one four-second trace for each scheduler. The first `1,000,000,000` cycles are warmup, and the profiles contain the remaining 330 workloads: 180 camera-detection requests, 60 lane-segmentation requests, and 90 driver-monitoring requests. Each run submitted 440 workloads and completed 53,400 kernels. The post-warmup kernel profiles contain 40,050 kernel invocations. Scheduler statistics below include only decisions at or after the warmup boundary.

The accelerator runs at 1 GHz and uses an eight-core `4x2` CCG mesh with four DMA tiles. Camera detection runs YOLOX-Nano for four 15 FPS camera streams with a 50 ms SLO. Lane segmentation runs Fast-SCNN at 20 FPS with a 40 ms SLO, and driver monitoring runs MobileNetV3-Small at 30 FPS with a 25 ms SLO. The Virtual policy partitions the mesh into four cores for camera detection, two for lane segmentation, and two for driver monitoring.

All four policies completed every measured workload without a deadline miss. The minimum slack remained above 24 ms, so deadline misses cannot distinguish the policies in this trace.

| Policy | Deadline misses | Mean response (ms) | p50 (ms) | p95 (ms) | p99 (ms) | Mean queueing (ms) | Mean execution (ms) | Minimum slack (ms) | Simulator wall time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Sequential | 0 / 330 | 0.708 | 0.548 | 1.587 | 1.587 | 0.224 | 0.484 | 24.832 | 4,380.6 |
| Preemptive | 0 / 330 | 0.742 | 0.700 | 1.587 | 1.587 | 0.022 | 0.720 | 24.832 | 4,358.1 |
| Virtual | 0 / 330 | 2.479 | 1.148 | 8.223 | 8.223 | 0.470 | 2.010 | 24.260 | 3,685.5 |
| CALM | 0 / 330 | 0.774 | 0.732 | 1.587 | 1.587 | 0.066 | 0.707 | 24.832 | 5,168.4 |

Sequential produced the lowest mean response time. Its mean was 4.6% lower than Preemptive's and 8.5% lower than CALM's. CALM's mean was 4.3% higher than Preemptive's. The three shared-device policies had the same overall p95 because the deterministic 1.587 ms lane-segmentation response occupied the upper tail of the combined workload distribution.

Virtual had the highest mean and tail latency. Its mean response was 3.50 times Sequential's, and its p95 was 5.18 times that of the shared-device policies. The fixed partitions still met every deadline because the SLOs were much longer than the observed execution times.

Queueing is measured as `start_cycle - arrival_cycle`, where `start_cycle` is the first kernel's actual start. The class-level results are:

| Policy | Camera response / p95 / queue / execution (ms) | Lane response / p95 / queue / execution (ms) | Driver response / p95 / queue / execution (ms) |
|---|---:|---:|---:|
| Sequential | 0.685 / 1.096 / 0.411 / 0.274 | 1.587 / 1.587 / 0.000 / 1.587 | 0.168 / 0.168 / 0.000 / 0.168 |
| Preemptive | 0.748 / 0.837 / 0.041 / 0.707 | 1.587 / 1.587 / 0.000 / 1.587 | 0.168 / 0.168 / 0.000 / 0.168 |
| Virtual | 1.435 / 2.296 / 0.861 / 0.574 | 8.223 / 8.223 / 0.000 / 8.223 | 0.740 / 0.740 / 0.000 / 0.740 |
| CALM | 0.806 / 0.897 / 0.122 / 0.684 | 1.587 / 1.587 / 0.000 / 1.587 | 0.168 / 0.168 / 0.000 / 0.168 |

Sequential serialized the four camera requests released at each camera period. Every request then executed in 0.274 ms, but its position in the burst increased mean camera queueing to 0.411 ms and camera p95 to 1.096 ms. Preemptive reduced camera queueing by 90.1% and camera p95 by 23.6%, although the simultaneous progress of multiple requests raised mean camera response by 9.2% relative to Sequential.

CALM reduced mean camera execution time from Preemptive's 0.707 ms to 0.684 ms, but increased the delay before the first kernel from 0.041 ms to 0.122 ms. Consequently, its camera mean response was 7.8% higher and its camera p95 was 7.1% higher than Preemptive's. The difference is an admission and scheduling effect rather than slower accumulated kernel service.

Lane and driver profiles are exactly identical under Sequential, Preemptive, and CALM. Their phased arrivals do not create contention with the camera bursts in this trace, so each executes as an isolated singleton and uses the common singleton placement policy. Virtual also eliminates their initial queueing, but the two-core partitions increase lane execution to 8.223 ms and driver execution to 0.740 ms.

The post-warmup scheduling decisions and core allocations were:

| Policy | Decisions | Kernels selected | Multi-kernel decisions | Cross-class decisions | Mean cores per kernel | Core-count stddev | Eight-core kernels | Camera / lane / driver mean cores |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Sequential | 40,050 | 40,050 | 0 (0.000%) | 0 | 5.336 | 2.177 | 30.26% | 5.417 / 5.333 / 5.065 |
| Preemptive | 39,915 | 40,050 | 90 (0.225%) | 0 | 3.983 | 1.953 | 13.18% | 3.487 / 5.333 / 5.065 |
| Virtual | 40,048 | 40,050 | 2 (0.005%) | 2 | 2.552 | 1.116 | 0.00% | 2.974 / 1.367 / 1.645 |
| CALM | 39,915 | 40,050 | 135 (0.338%) | 0 | 3.960 | 1.948 | 13.18% | 3.455 / 5.333 / 5.065 |

Sequential selected one kernel per decision. Preemptive produced 45 two-camera and 45 three-camera decisions by independently placing multiple ready kernels in one scheduling pass; these records are not CALM-style joint plans and report zero predicted benefit.

CALM produced 135 two-camera joint decisions and no triple or cross-class decision. Its mean core allocation was only 0.6% lower than Preemptive's, and its camera allocation was 0.9% lower. The common singleton policy therefore removed the large allocation difference between the two policies; the remaining difference comes from CALM's selected joint camera plans.

The 135 CALM joint decisions consisted of three repeated plan forms, each selected 45 times. Their predicted benefits were 40.0%, 20.0%, and 10.07%, giving a mean positive benefit of 23.36% and a median of 20.0%. Joint decisions represented only 0.338% of all measured decisions, so mean predicted benefit across every CALM decision was 0.0790%. The trace exercises a small number of repeated same-class camera pairs rather than broad heterogeneous co-location.

The two Virtual multi-kernel decisions paired lane and driver kernels in separate fixed domains. They represent concurrent dispatch across isolated partitions, not dynamic sharing of one core pool.

The kernel profiles aggregate invocations with the same kernel type and tensor shapes. Latency is weighted by invocation count, and core-count variance is pooled across profile entries.

| Policy | Profiled kernels | Weighted mean kernel latency (us) | Sum of kernel latencies (ms) | Mean cores | Core-count stddev |
|---|---:|---:|---:|---:|---:|
| Sequential | 40,050 | 3.985 | 159.589 | 5.336 | 2.177 |
| Preemptive | 40,050 | 5.496 | 220.126 | 3.983 | 1.953 |
| Virtual | 40,050 | 16.561 | 663.265 | 2.552 | 1.116 |
| CALM | 40,050 | 5.366 | 214.924 | 3.960 | 1.948 |

| Policy | Camera mean kernel latency (us) | Lane mean kernel latency (us) | Driver mean kernel latency (us) |
|---|---:|---:|---:|
| Sequential | 1.756 | 26.444 | 1.801 |
| Preemptive | 3.912 | 26.444 | 1.801 |
| Virtual | 3.679 | 137.053 | 7.953 |
| CALM | 3.727 | 26.444 | 1.801 |

CALM accumulated 0.581 ms of kernel latency per camera request, 4.7% less than Preemptive's 0.610 ms. The intervals between a request's kernels contributed another 0.103 ms under CALM and 0.097 ms under Preemptive. CALM therefore retained a shorter first-start-to-completion interval, but its 0.081 ms increase in initial queueing outweighed the 0.023 ms execution reduction and produced the higher response time.

The largest shared-device kernel contributors were all stable across Sequential, Preemptive, and CALM. The initial lane convolution averaged 354.723 us and contributed 21.283 ms across 60 requests. The final lane upsample MemCopy averaged 294.934 us and contributed 17.696 ms, while the `64x128x384` depthwise downsampling convolution averaged 249.500 us and contributed 14.970 ms.

The camera input MemCopy varied with concurrency: 66.576 us under Sequential, 95.278 us under Preemptive, 90.501 us under CALM, and 208.022 us under Virtual. Under Virtual, the initial lane convolution increased to 1.710 ms and contributed 102.589 ms, while the final lane upsample increased to 1.049 ms and contributed 62.915 ms. These kernels explain a substantial portion of the Virtual lane slowdown.

Simulator wall time measures Python planning, materialization, and event processing rather than simulated accelerator latency. Virtual was fastest at 3,685.5 seconds. Preemptive and Sequential required 4,358.1 and 4,380.6 seconds. CALM was slowest at 5,168.4 seconds, 18.6% slower than Preemptive and 40.2% slower than Virtual. The profiles do not separate joint-plan search cost from the remaining simulator work, but CALM's additional candidate evaluation is consistent with this overhead.

### Qualitative Comparison of Scheduler Variants

**Sequential** achieved the lowest mean response because each short workload received broad access to the mesh and completed before the next non-camera release. Its weakness appears in synchronized camera bursts: later requests wait for earlier requests, producing the highest camera p95 despite the shortest per-request execution time.

**Preemptive** provided the best camera tail latency and the lowest mean response among the concurrent shared-device policies. Kernel-boundary interleaving reduced camera admission delay without changing the isolated lane and driver paths. It used fewer cores per camera kernel than Sequential and incurred longer individual request execution, but it prevented the long burst-position delay seen under Sequential.

**Virtual** provided fixed spatial isolation and the shortest simulator wall time. The `4/2/2` partition was inefficient for the measured service demands. Lane and driver started immediately but ran much longer on their restricted domains, while the four camera streams still serialized within the camera partition. Isolation delivered no SLO advantage because all shared-device policies already had large slack.

**CALM** selected 135 joint camera pairs and achieved slightly lower accumulated camera kernel latency and execution time than Preemptive. It nevertheless delayed the first kernel of camera requests more often, resulting in worse camera mean and tail response. Its core allocation was nearly identical to Preemptive after singleton placement was unified, and lane and driver behavior was exactly identical. CALM never formed a cross-class plan in this trace.

The result supports kernel-boundary interleaving for synchronized camera requests: Preemptive reduced camera p95 by 23.6% relative to Sequential. It does not establish an advantage for CALM's joint placement search. CALM improved neither deadline misses nor mean or tail response relative to Preemptive, used joint planning in only 0.338% of measured decisions, and incurred the highest simulator wall time.

The workload remains underloaded relative to its SLOs. The worst response was 8.223 ms under Virtual, while the shortest deadline was 25 ms. A stronger CALM evaluation requires overlapping camera, lane, and driver releases, enough sustained contention to create repeated cross-class choices, and invocation-level measurements of observed overlap and joint makespan. Those conditions are necessary to determine whether predicted joint-plan benefits translate into workload-level latency or deadline improvements.
