# Research Overview

## Terminology

### 기본 용어 정리

| Term | Abbr. | Description |
| --- | --- | --- |
| Single Core NPU | - | 단일 코어만을 포함하는 NPU 아키텍처 |
| Multi-Core NPU | - | 두개 이상의 코어들과 공유 버퍼(GLB)를 포함하는 NPU 아키텍처 |
| Multi-Tile Accelerator | MTA | 여러 개의 연산 타일 및 DMA 엔진이 NoC로 연결된 가속기 (NPU) 아키텍처 |
| Systolic Array | SA | 시스톨릭 배열 |
| Global Buffer | GLB | 멀티 코어 NPU에서 코어 간 서로 공유하는 버퍼 |
| Local Buffer | LBUF | 멀티 코어 NPU에서 코어가 소유하는 로컬 버퍼로 싱글 코어에서 UBUF와 동일 |
| Unified Buffer | UBUF | 싱글 코어 NPU에서 코어가 소유하는 로컬 버퍼로, IFM/WGT/PSUM이 전부 다 저장될 수 있음 |
| Accumulation Buffer | ACCB | SA에서 출력된 PSUM이 누산되는 버퍼 |
| Input Feature Map | IFM | - |
| Weight | WGT | - |
| Output Feature Map | OFM | - |
| Matrix Multiplication Unit | MXU | - |
| Vector Processing Unit | VPU | - |

### 딥러닝 워크로드 용어 정리

| Term | Abbr. | Korean | Description |
| --- | --- | --- | --- |
| Workload | - | 워크로드 | 가속기에 dispatch되는 작업의 단위로, 주로 DNN 모델이 이에 해당함
| Job | - | - | 가속기에 dispatch되는 작업의 단위로, DNN 모델 혹은 그보다 작은 sub-model 수준의 워크로드를 지칭함
| Layer | - | 계층 | DNN 모델 그래프를 구성하는 계층(혹은 노드)으로, 본 문서에서는 LLM의 구성 단위를 의미하지 않음
| Operator | - | 연산 | DNN 모델 그래프를 구성하는 최하위 계층으로, pytorch BLAS 라이브러리에서 제공하는 primitive들이 이에 해당함
| Kernel | - | 커널 | 컴파일러에 의해 변환된 가속기에서 실행 가능한 연산을 의미함
| Arithmetic Intensiry | AI | 연산 강도 | - 


## Background and Motivation

### 싱글 코어 NPU 멀티테넌시 

싱글 코어 NPU는 전통적인 TPU-like 아키텍처를 의미한다. 일반적으로 싱글 코어 NPU는 double buffering을 위한 Unified 버퍼(UBUF)와 Accumulation 버퍼(ACCB), 그리고 PE 배열 (SA)로 이루어진다. 이때, UBUF는 프로그래머 혹은 컴파일러에 의해 직접적으로 접근 가능한 메모리라기보다는 메모리에서 로드되는 데이터를 prefetch하기 위한 버퍼로 활용된다. 연산 결과는 ACCB에 저장되었다가 off-chip DRAM으로 커밋되며, 필요한 경우 ACCB의 데이터가 UBUF로 옮겨져 벡터 유닛(VPU)에서 활용된다.

**싱글 코어 NPU에서의 멀티테넌시 지원은 통상적으로 선점형(preemptive) 스케줄링을 통해 지원**하는 경우가 많다. 워크로드를 스케줄링하는 관점에서 선점형(preemptive) 스케줄링은 비선점형(non-preemptive) 스케줄링에 비해 응답시간(response time), fairness, 그리고 SLO 만족도 측면에서 이점을 가질 수 있다. 특히 서로 독립적인 여러 워크로드가 동일한 연산 자원을 공유하는 경우, 장시간 실행되는 워크로드가 짧거나 높은 우선순위를 갖는 워크로드를 장시간 blocking하는 현상을 방지할 수 있다. [[2]] 또한, co-location되는 워크로드가 서로 heterogeneous한 특성을 보이는 경우, 전체적인 리소스 활용율이 증가하여 전체적인 지연시간이 감소하기도 한다. [[4]]

하지만, 선점형 스케줄링은 필연적으로 context switching을 수반한다. 즉, 두개 이상의 워크로드를 선점형으로 실행하는 경우 이전에 실행하던 워크로드에 대한 context를 미리 off-chip DRAM에 checkpoint하고, 나중에 해당 워크로드를 실횅하기 전 context를 restore해야 한다. 이때, SPM에 저장된 모든 데이터가 반드시 context switching 시 DRAM으로 flush되어야 하는 것은 아니다. Context에 포함되는지 여부는 해당 데이터가 다른 위치에서 다시 복구 가능한지에 따라 결정된다.

예를 들어 weight나 DRAM에서 prefetch한 input tile과 같이 원본 데이터가 off-chip memory에 존재하는 read-only 데이터의 경우, context switching 시 이를 버린 뒤 해당 task가 다시 실행될 때 재로드할 수 있다. 반면 아직 DRAM에 write-back되지 않은 intermediate output이나 partial result처럼 최신 상태가 on-chip SPM에만 존재하는 경우에는 해당 state를 보존해야 한다. 따라서 NPU의 context는 단순히 PE 내부 register와 accumulation buffer만으로 정의되는 것이 아니라, **현재 실행을 재개하기 위해 보존해야 하는 모든 live on-chip state**로 보는 것이 더 정확하다. 128×128 systolic array를 예로 들어 각 PE에 FP32 partial sum 하나가 존재한다고 가정하면 accumulator state만으로도

$$
128 \times 128 \times 4B = 64\text{KB}
$$

의 context가 발생할 수 있다. 여기에 weight, input, pipeline state, 혹은 SPM에 저장된 live intermediate activation까지 포함하면 실제 context의 크기는 더 커질 수 있다. 반대로 weight나 prefetched input과 같이 재로드 가능한 데이터는 checkpoint하지 않고 버릴 수 있으므로 context size에서 제외될 수 있다. 다만 이러한 데이터는 task 복원 시 다시 DRAM에서 fetch해야 하므로 context switching으로 인해 추가적인 DRAM traffic이 발생할 수 있다.

PREMA [[2]]는 싱글 코어 NPU에서 선점형 스케줄링을 제안하였으며, 해당 논문에서도 context switching 오버헤드에 대해 언급하고 있다. 논문에서는 최악의 경우 UBUF와 ACCB 전체를 checkpoint할 때, 약 59us 정도의 오버헤드가 있었다고 하였으며, 이는 전체적인 추론 시간 대비 2.6% 수준으로 짧다고 언급한다. V10 [[4]] 은 context switching이 발생할 때 PE에 저장된 partial sum을 context로 저장하지 않고 input을 대신 context로 저장한다. 이후 partial sum은 recompute되므로 별도의 PE flush 과정이 없어지면서 preemption된 다음번 워크로드는 연속적으로 처리된다. 논문에 따르면 이러한 기법을 통해 context switching 오버헤드를 128x128 SA 기준으로 384 사이클로 크게 줄일 수 있었다고 한다. Layerweaver [[3]] 또한 커널 수준의 preemption을 하기 때문에 이 과정에서 context switching이 일어날 수 있지만, 이를 방지하기 위해 ACCB의 크기를 1.5MB만큼 늘려서 강제로 off-chip checkpoint 과정이 일어나지 않도록 한다.

AI-MT [[6]]는 다른 싱글 코어 NPU 스케줄링 논문과 다르게 선점형 스케줄링을 사용하지 않는다. 다만, 스케줄링을 하기 전 워크로드를 구성하는 모든 계층(혹은 커널)을 sublayer로 나누고, 이를 단위로 스케줄링한다. 따라서 context checkpoint에 의한 오버헤드는 발생하지 않지만, 특정 sublayer를 위한 데이터를 너무 많이 SRAM이 prefetch한 경우, 해당 커널을 우선적으로 스케줄링하여 최대한 빠르게 SRAM이 release될 수 있도록 한다. 이로써 SRAM capacity가 부족하여 다른 커널이 지나치게 길게 대기하는 경우를 방지한다.

> MAGMA [[7]]은 AI-MT 논문을 homogeneous multiple systolic array에 대응하는 스케줄러 논문으로 소개하고 있지만, 실제 논문을 읽어보면 AI-MT 스케줄러가 코어에 내장되므로 싱글 코어 멀티테넌시 논문으로 분류한다.

### 멀티 코어 NPU 멀티테넌시

멀티 코어 NPU는 여러 개의 코어를 포함하는 NPU 아키텍처로, 싱글 코어 아키텍처에 더해 코어들을 연결하는 NoC와 이들이 공유하는 Global 버퍼(GLB)를 포함한다. 컴파일러 혹은 런타임은 GLB에 데이터를 넣고 NPU 코어는 GLB에서 데이터를 읽어서 로컬 버퍼(LBUF, 싱글 코어 구성에서 UBUF와 동일)에 넣는다. 이후 싱글 코어에서와 동일하게 연산을 수행한 후 연산 결과를 GLB에 보내고, 동기화 한 후 GLB에서 최종적으로 off-chip DRAM으로 연산 결과가 저장된다.

앞서 언급했던 싱글 코어에서의 멀티테넌시 기법들은 멀티 코어 NPU에서도 동일하게 적용 가능하다. 다만, 이 경우 전체적인 context switching 오버헤드가 코어의 개수에 비례해서 선형적으로 증가하므로 이 부분은 감안해야 한다. AI-MT의 경우 context swtiching 오버헤드가 딱히 없고, 스케줄러가 코어 마다 존재하므로 무리 없이 멀티 코어 NPU 아키텍처로 확장이 가능하다.

멀티 코어 NPU에서 적용할 수 있는 또다른 스케줄링 기법은 **두개 이상의 워크로드에 대한 spatial co-location**이다. 이를 지원하기 위해서는 컴파일러가 단일 코어 수준에서 컴파일 한 job을 런타임이 동적으로 placement 할 수 있어야 한다. 가장 단순하면서 직관적인 방법은 **reconfigurable interconnect network를 활용하여 가속기 구조를 여러 개의 물리적 인스턴스로 재구성하는 것**이다. 예를 들어 Planaria [[1]]은 각 워크로드가 최적으로 실행될 수 있는 가속기 dimension이 각기 다르다는 점에 착안하여, 재구성 가능한 sub-accelerator들을 reconfigurable NoC로 연결한 형태로 설계하였으며, 런타임은 다양한 가속기 구성을 통해 컴파일 된 여러 워크로드를 동시에 가속기에 매핑한다. 이외에도 다양한 reconfigurable interconnect 기반의 가속기가 spatial 멀티테넌시를 지원한다. [[11]] [[12]]

또다른 방법은 **멀티코어 NPU에서 일부 코어를 그룹화하여 가상 가속기 인스턴스를 생성하는 이른바 가상화를 지원**하는 것이다. 예를 들어 Neu10 [[9]]은 OS(Hypervisor)의 도움을 통해 NPU를 가상화하는 솔루션을 제안하였으며, spatial isolation과 temporal context switching을 지원하는 MXU/VPU 명령어 스케줄링 알고리즘을 제안한다. vNPU [[10]]는 이에 더해 mesh interconnect의 topology를 고려한 멀티 코어 NPU 가상화 기법을 제안한다.

하지만, 가속기 재구성이나 가상화와 같이 정적으로 가속기 인스턴스를 생성하는 방법은 워크로드의 특성은 반영할 수 있으나, **커널(혹은 계층) 간 이질적 연산 특성은 반영하지 못한다**. 딥러닝 워크로드를 구성하는 커널들은 서로 각기 다른 연산 강도를 가지고 있으나, 가속기 인스턴스를 생성할 때는 최악의 경우를 상정하여 연산기와 DMA 채널을 수를 할당해야 한다. 따라서 워크로드를 구성하는 여러 커널들 중 연산 강도가 높거나 낮은 커널들은 연산기 및 메모리 병목으로 인해 상대적으로 긴 지연시간을 보일 수 있다. 물론 Neu10 [[9]]의 경우 context switching을 통한 temporal time multiplexing을 지원하므로, 이러한 리소스 활용율 문제는 완화할 수 있으나, context switching에 의한 오버헤드가 추가적으로 발생할 가능성이 있다.

이러한 문제를 해결하는 또다른 방법은 **커널 수준의 멀티 코어 스케줄링을 활용하는 것**이다. 이 기법은 각 워크로드가 실행되는 가속기 인스턴스를 제한하지 않고, 상황에 따라 적절하게 커널을 여러 코어에 할당하는 것이다. 예를 들어 Herald [[13]]는 각 가속기 코어에 커널을 동적으로 할당하는 스케줄링 알고리즘을 제안하였으며, 이 과정에서 가속기 코어의 heterogeneous dataflow를 반영하도록 하였다. 또한, MAGMA [[14]] 는 Herald에서 제안하는 스케줄링 알고리즘이 정적인 휴리스틱을 기반으로 하고 있다는 점을 지적하면서 GA(Genatic Algorithm, 유전적 알고리즘)를 기반으로 최적화 문제를 재해석하였다.

이러한 기존 스케줄링 알고리즘은 

### GPU에서의 커널 수준 스케줄링

TBD

### MTA에서의 멀티테넌시 지원 시 이슈

TBD

<!-- REFERENCE -->

[1]: https://ieeexplore.ieee.org/document/9251939                   "Planaria"
[2]: https://ieeexplore.ieee.org/document/9065590                   "PREMA"
[3]: https://ieeexplore.ieee.org/document/9407236                   "Layerweaver"
[4]: https://dl.acm.org/doi/10.1145/3579371.3589059                 "V10"
[5]: https://arxiv.org/abs/2305.05843                               "MoCA"
[6]: https://ieeexplore.ieee.org/document/9138929/                  "AI-MT"
[7]: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9773188   "MAGMA"
[8]: https://dl.acm.org/doi/10.1145/3352460.3358302                 "Simba"
[9]: https://dl.acm.org/doi/10.1109/MICRO61859.2024.00011           "Neu10, MICRO'24"
[10]: https://dl.acm.org/doi/10.1145/3695053.3731002                "vNPU, ISCA'25"
[11]: https://dl.acm.org/doi/10.1145/3296957.3173176                "MAERI, ASPLOS'18"
[12]: https://ieeexplore.ieee.org/document/9065523                  "SIGMA, HPCA'20"
[13]: https://dl.acm.org/doi/abs/10.1109/LES.2024.3443628           "Herald, HPCA'21"
[14]: https://ieeexplore.ieee.org/document/9773188                  "MAGMA, HPCA'22"