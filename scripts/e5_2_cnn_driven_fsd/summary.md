## Experiment: `e5_2_cnn_driven_fsd`

### Quantitative Analysis of Scheduling Efficiency

The current `run.py` compares Sequential, Preemptive, Spatial, and Virtual scheduling. Each profile covers a four-second request trace at a 1 GHz cycle rate. Workloads arriving before cycle `1,000,000,000` are excluded from workload and kernel measurements, leaving 330 requests: 180 camera-detection, 60 lane-segmentation, and 90 driver-monitoring requests. Each run submitted 440 requests and completed 53,400 kernels; the measured requests account for 40,050 kernel invocations. Scheduler statistics below include only decisions at or after the warmup boundary.

The accelerator has an eight-core `4x2` CCG mesh and four DMA tiles. Four YOLOX-Nano camera streams arrive together at 15 FPS per stream and have a 50 ms SLO. Fast-SCNN lane segmentation arrives at 20 FPS with a 5 ms phase and a 40 ms SLO. MobileNetV3-Small driver monitoring arrives at 30 FPS with a 10 ms phase and a 25 ms SLO. Virtual assigns four cores and two DMA tiles to camera detection, two cores and one DMA tile to lane segmentation, and two cores and one DMA tile to driver monitoring.

Response time is `completion_cycle - arrival_cycle`; queueing is `start_cycle - arrival_cycle`; execution is `completion_cycle - start_cycle`. Percentiles use linear interpolation over the sorted request times. All four policies met every measured deadline, so response time and resource use distinguish them more clearly than deadline misses in this trace.

| Policy | Deadline misses | Mean response (ms) | p50 (ms) | p95 (ms) | p99 (ms) | Mean queueing (ms) | Mean execution (ms) | Minimum slack (ms) | Simulator wall time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Sequential | 0 / 330 | 1.980 | 1.889 | 3.778 | 3.778 | 0.773 | 1.207 | 24.579 | 9,564.9 |
| Preemptive | 0 / 330 | 3.190 | 4.580 | 4.581 | 4.581 | 0.112 | 3.078 | 24.579 | 11,006.2 |
| Spatial | 0 / 330 | 2.092 | 2.466 | 3.172 | 3.172 | 0.049 | 2.043 | 24.579 | 11,390.1 |
| Virtual | 0 / 330 | 5.125 | 3.011 | 15.345 | 15.345 | 1.232 | 3.893 | 23.967 | 10,307.2 |

Sequential had the lowest mean response. Spatial's mean was 5.7% higher, but its overall p95 was 16.0% lower because concurrent camera placement shortened the camera burst's tail. Preemptive's mean was 61.2% higher than Sequential's. Virtual's mean was 2.59 times Sequential's, and its p95 was dominated by lane segmentation on its two-core domain.

The class-level response, queueing, and execution times show where these differences arise:

| Policy | Camera response / p95 / queue / execution (ms) | Lane response / p95 / queue / execution (ms) | Driver response / p95 / queue / execution (ms) |
|---|---:|---:|---:|
| Sequential | 2.361 / 3.778 / 1.417 / 0.944 | 3.172 / 3.172 / 0.000 / 3.172 | 0.421 / 0.421 / 0.000 / 0.421 |
| Preemptive | 4.580 / 4.581 / 0.206 / 4.375 | 3.172 / 3.172 / 0.000 / 3.172 | 0.421 / 0.421 / 0.000 / 0.421 |
| Spatial | 2.568 / 2.924 / 0.090 / 2.478 | 3.172 / 3.172 / 0.000 / 3.172 | 0.421 / 0.421 / 0.000 / 0.421 |
| Virtual | 3.764 / 6.022 / 2.258 / 1.505 | 15.345 / 15.345 / 0.000 / 15.345 | 1.033 / 1.033 / 0.000 / 1.033 |

Sequential processes each synchronized four-camera burst one request at a time. Every camera request executes in about 0.944 ms, while the last request waits for the earlier three. Spatial cuts mean camera queueing from 1.417 to 0.090 ms (93.6%) and camera p95 from 3.778 to 2.924 ms (22.6%). Concurrent placement lengthens camera execution to 2.478 ms, so its mean camera response is 8.8% above Sequential's despite the shorter tail.

Preemptive also reduces initial camera queueing, to 0.206 ms, but the interval from first kernel start to request completion rises to 4.375 ms. Its camera mean and p95 both reach about 4.581 ms. Spatial lowers camera mean response by 43.9% and p95 by 36.2% relative to Preemptive. Lane and driver response profiles are identical under Sequential, Preemptive, and Spatial: their phased releases do not contend with the camera bursts in the measured trace.

Virtual isolates the three classes, but its four camera streams still queue within the camera domain. The two-core lane and driver domains increase their execution times to 15.345 and 1.033 ms, respectively. They start immediately, yet the lane slowdown sets the overall p95. The smallest measured slack, 23.967 ms, belongs to a Virtual driver request; it is still far from missing its deadline.

Post-warmup scheduler decisions and actual core allocations were:

| Policy | Decisions | Kernels selected | Multi-kernel decisions | Cross-class decisions | Mean cores per kernel | Core-count stddev | Eight-core kernels | Camera / lane / driver mean cores |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Sequential | 40,050 | 40,050 | 0 | 0 | 6.112 | 3.043 | 71.69% | 6.205 / 6.067 / 5.817 |
| Preemptive | 40,050 | 40,050 | 0 | 0 | 6.058 | 3.051 | 70.34% | 6.128 / 6.067 / 5.817 |
| Spatial | 39,870 | 40,050 | 135 (0.339%) | 0 | 3.716 | 2.601 | 22.81% | 2.788 / 6.067 / 5.817 |
| Virtual | 40,050 | 40,050 | 0 | 0 | 2.769 | 1.330 | 0.00% | 3.231 / 1.550 / 1.742 |

Sequential, Preemptive, and Virtual each recorded one kernel per scheduling decision. Spatial recorded 90 two-camera decisions and 45 three-camera decisions; the other 39,735 decisions selected one kernel. No Spatial joint decision combined different workload classes. Its mean camera allocation was 2.788 cores, compared with more than six under Sequential and Preemptive. Lane and driver allocations were identical across these three shared-device policies.

The 135 Spatial joint decisions repeat three plan forms, each 45 times. Their predicted benefits were 66.67% for the three-kernel plan and 48.92% and 58.11% for the two-kernel plans. The mean predicted benefit among selected joint plans was 57.90%. This is the scheduler's estimate of joint makespan savings relative to estimated sequential service, not a measured reduction in request response time. Joint decisions represented only 0.339% of measured decisions, and all occurred within synchronized camera bursts.

Kernel profiles aggregate invocations with the same kernel type and tensor shapes. The table weights entry latency by invocation count and pools core-count variance across entries. A sum of kernel latencies adds all measured invocations, including kernels that run concurrently, so it is not a workload completion time.

| Policy | Profiled kernels | Weighted mean kernel latency (us) | Sum of kernel latencies (ms) | Mean cores | Core-count stddev |
|---|---:|---:|---:|---:|---:|
| Sequential | 40,050 | 9.944 | 398.257 | 6.112 | 3.043 |
| Preemptive | 40,050 | 10.846 | 434.390 | 6.058 | 3.051 |
| Spatial | 40,050 | 16.834 | 674.214 | 3.716 | 2.601 |
| Virtual | 40,050 | 32.077 | 1,284.668 | 2.769 | 1.330 |

| Policy | Camera mean kernel latency (us) | Lane mean kernel latency (us) | Driver mean kernel latency (us) |
|---|---:|---:|---:|
| Sequential | 6.054 | 52.872 | 4.531 |
| Preemptive | 7.341 | 52.872 | 4.531 |
| Spatial | 15.882 | 52.872 | 4.531 |
| Virtual | 9.650 | 255.757 | 11.107 |

Per camera request, the summed kernel latencies were 0.944 ms for Sequential, 1.145 ms for Preemptive, 2.478 ms for Spatial, and 1.505 ms for Virtual. Under Preemptive, the 4.375 ms first-start-to-completion interval exceeds the 1.145 ms kernel sum by about 3.230 ms, indicating substantial time between a request's kernel executions. The corresponding Spatial interval and kernel sum both round to 2.478 ms. These aggregates describe why Preemptive's low initial queueing does not translate into low camera response time.

The initial lane convolution (`1x512x1024x3`) averaged 831.763 us and contributed 49.906 ms over 60 requests under each shared-device policy. The `1x64x128x384` depthwise convolution contributed another 37.462 ms at 624.360 us per invocation. Spatial's initial camera convolution (`1x208x208x12`) rose to 538.989 us per invocation and contributed 97.018 ms across camera requests, versus 154.389 us and 27.790 ms under Sequential. Its camera input MemCopy averaged 338.600 us, versus 136.266 us under Sequential. These slower camera kernels accompany the smaller shared core placements.

Virtual's initial lane convolution averaged 2,717.181 us and contributed 163.031 ms; its final lane MemCopy averaged 1,359.393 us and contributed 81.564 ms. These large per-kernel costs on the restricted lane domain help explain its 15.345 ms lane response.

Simulator wall time measures the host-side simulation, including planning and event processing, rather than simulated accelerator latency. Sequential took 9,564.9 s, Virtual 10,307.2 s, Preemptive 11,006.2 s, and Spatial 11,390.1 s. Spatial was 19.1% slower than Sequential in wall time. The profiles do not isolate joint-plan search cost, and runs executed in parallel by default, so these timings should not be treated as a controlled scheduler-overhead benchmark.

### Qualitative Comparison of Scheduler Variants

**Sequential** achieved the lowest overall and camera mean response by giving each short camera request broad access to the mesh until completion. Synchronized camera arrivals create its main weakness: later requests wait behind earlier ones, producing a 3.778 ms camera p95.

**Preemptive** admitted camera requests sooner than Sequential, but kernel-boundary interleaving stretched each camera request after its first kernel began. It had the highest camera mean and p95 of the three shared-device policies, despite using nearly the same mean number of camera cores as Sequential.

**Spatial** jointly placed camera kernels in 135 decisions. It achieved the best camera p95 and the lowest camera queueing, while its smaller core placements increased individual kernel and request execution time. It did not improve the overall mean response over Sequential, and its trace contains no cross-class joint placement.

**Virtual** kept camera, lane, and driver requests in fixed domains. Lane and driver started without queueing, but their two-core domains greatly increased service time. Its overall mean and p95 were the highest, while all requests still met their SLOs.

In this trace, Spatial trades longer camera execution for a shorter camera tail; Sequential remains best on mean response, and none of the policies differ on deadline misses. The phased lane and driver releases leave the experiment dominated by synchronized, same-class camera bursts. The observed results therefore support a camera-tail benefit from Spatial joint placement under these arrivals, but do not measure how it would behave under sustained cross-class contention.
