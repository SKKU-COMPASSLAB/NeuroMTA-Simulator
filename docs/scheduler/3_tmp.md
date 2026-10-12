제안하는 방법은 각 커널의 연산 타일 점유량, NoC 대역폭 요구량, DMA 채널 대역폭 요구량을 함께 모델링하고, 이를 기반으로 여러 커널의 실행 위치와 `core mesh` 크기를 결정하는 방식이다.

현재 가정하는 MTA 아키텍처에서는 단순한 Resource Partitioning이나 Bin Packing보다 Topology-Aware, Bandwidth-Aware Spatial Scheduling으로 문제를 정의하는 것이 적절해 보인다.

특히 중요한 것은 다음 세 가지다.

1. Bandwidth-Aware Co-location: 커널 간 대역폭 요구량이 상호 보완적인 조합을 찾는다.
2. Topology-Aware Placement: 동일한 커널이라도 `core mesh`의 위치와 형태에 따라 NoC contention이 달라진다는 점을 고려한다.
3. Dynamic Resource Allocation: 사용 가능한 리소스에 따라 각 커널의 `core mesh` 크기를 동적으로 조절한다.

이를 구현하기 위해 세 가지 접근 방법을 제안하겠다.

| 방법                            | 핵심 아이디어                                     | 구현 난이도 |
| ----------------------------- | ------------------------------------------- | ------ |
| 1. Resource Vector 기반 Greedy  | Compute, intra/outer-mesh, DMA 대역폭을 벡터로 모델링 | 낮음     |
| 2. NoC Link-Aware Greedy      | 실제 Mesh NoC의 링크별 트래픽과 병목을 예측                | 중간     |
| 3. Optimization 기반 Scheduling | MILP/CP-SAT으로 커널 조합 및 배치를 공동 최적화            | 높음     |

우선 추천하는 것은 2번이다. 1번보다 실제 아키텍처에 잘 대응하면서, 3번보다 런타임 스케줄러로 구현하기가 수월하다. 또한 NumPy를 활용해 대부분의 대역폭 계산을 벡터화할 수 있다.

## 1. 스케줄링 문제의 수학적 모델

### 1.1 하드웨어 리소스 모델

MTA를 다음과 같이 정의하자.

- \\(H\times W\\): 전체 연산 타일 Grid 크기
- \\(N=H\times W\\): 전체 연산 타일 개수
- \\(D\\): DMA 채널 개수
- \\(B\_{\text{NoC}}\\): NoC의 각 물리 링크가 지원하는 최대 대역폭
- \\(B\_{\text{DMA}}[d]\\): DMA 채널 \\(d\\)의 최대 메모리 대역폭

예를 들어 88개 연산 타일을 가진 MTA라면 \\(H=8,W=11\\)로 설정할 수 있다.

여기서 중요한 것은 NoC 대역폭은 단일한 공유 리소스가 아니라, 여러 물리 링크의 집합이라는 점이다.

8×11 MTA의 논리적 구성 예시

단순화된 예시. 각 사각형은 연산 타일이며, 선은 NoC 링크를 의미한다. DMA 위치와 파티션 크기는 설명을 위한 가정이다.

예를 들어 두 커널이 서로 다른 연산 타일을 점유하더라도, DMA에서 데이터를 가져오는 경로가 겹친다면 동일한 NoC 링크에서 contention이 발생할 수 있다.

따라서 연산 타일의 점유 여부와 별개로 각 NoC 링크에 대한 사용량을 추적해야 한다.

### 1.2 커널 리소스 모델

커널 \\(k\\)의 배치를 다음과 같이 정의한다.

\\[ P_k=(x_k,y_k,h_k,w_k) \\]

여기서 \\((x_k,y_k)\\)는 `core mesh` 시작 좌표이며, \\(h_k,w_k\\)는 할당된 연산 타일 Grid 크기이다.

각 배치 후보에 대하여 다음 리소스 정보를 계산한다.

\\[ R_k(P_k)= \left( C_k,\\, L_k,\\, D_k \right) \\]

- \\(C_k\\): 점유하는 연산 타일의 Boolean Mask
- \\(L_k\\): NoC의 모든 물리 링크에 대한 예상 대역폭 사용량 벡터
- \\(D_k\\): 각 DMA 채널에 대한 예상 메모리 대역폭 사용량 벡터

두 커널이 동시에 실행되려면 다음 조건을 만족해야 한다.

연산 타일 점유 조건

\\[ C_i\cap C_j=\emptyset \\]

NoC 대역폭 조건

\\[ \sum_k L_k[\ell]\le B\_{\text{NoC}}[\ell] \quad \forall \ell \\]

DMA 채널 대역폭 조건

\\[ \sum_k D_k[d]\le B\_{\text{DMA}}[d] \quad \forall d \\]

핵심은 NoC 대역폭 조건을 전체 링크 사용량의 합이 아니라, 각 링크에 대한 개별 제약조건으로 적용한다는 것이다.

또한 모든 대역폭은 GB/s 등 동일한 단위를 사용하고, DMA에서 이동하는 데이터가 NoC를 통과한다면 해당 트래픽을 NoC 링크와 DMA 채널 양쪽에 각각 반영해야 한다. 이는 서로 다른 물리 리소스를 점유하기 때문에 중복 계산이 아니다.

이 모델은 각 커널이 요구하는 대역폭을 제공받을 수 있다는 조건에서 유효하다. 예측된 대역폭 요구량이 링크 또는 DMA 용량을 초과하면 실행 자체가 불가능하다는 의미는 아니며, 목표 성능을 유지하는 동시 실행이 불가능할 수 있다는 의미로 해석하는 것이 정확하다.

## 2. 방법 1: Resource Vector 기반 Greedy Scheduling

가장 간단한 구현 방법은 커널의 리소스 요구량을 NumPy 벡터로 표현하는 것이다.

커널 \\(k\\)의 리소스 벡터를 다음과 같이 정의할 수 있다.

\\[ R_k= \begin{bmatrix} N\_{\text{tiles}} & B\_{\text{intra}} & B\_{\text{outer}} & B\_{\text{DMA},0} & \cdots & B\_{\text{DMA},D-1} \end{bmatrix} \\]

예를 들어 2개의 DMA 채널을 사용하는 아키텍처라면 다음과 같이 표현한다.

```
import numpy as np# [tiles, intra BW, outer BW, DMA0 BW, DMA1 BW]capacity = np.array([88, 100, 100, 128, 128])linear    = np.array([16, 60, 70,  80,  0])reduction = np.array([ 8, 65, 10,   5,  0])element   = np.array([ 4,  5,  5,   0,  5])used = np.zeros_like(capacity)def can_colocate(demand):    return np.all(used + demand <= capacity)
```

이 방식의 장점은 여러 커널의 리소스 요구량을 단순 벡터 연산으로 처리할 수 있다는 점이다.

다만 여기서 `intra BW`와 `outer BW`에 별도 상한을 부여하는 것은 실제 하드웨어 자원을 추상화한 근사 모델이다. 두 종류의 트래픽이 동일한 물리 NoC 링크를 공유한다면 이를 독립적인 대역폭 자원으로 해석해서는 안 된다.

따라서 이 방식은 커널 조합을 빠르게 필터링하기 위한 1차 스케줄러로 사용하는 것이 적합하다.

## 3. 방법 2: NoC Link-Aware Greedy Scheduling

추천하는 방식이다.

### 3.1 NoC Link Utilization Matrix

NoC의 각 링크에 대한 대역폭 사용량을 NumPy 배열로 저장한다.

```
# H, W = 8, 11# direction: East, West, South, Northlink_usage = np.zeros((8, 11, 4))link_capacity = np.full((8, 11, 4), 48.0)# DMA channel bandwidthdma_usage = np.zeros(2)dma_capacity = np.array([128.0, 128.0])# Compute tile occupancycore_occupancy = np.zeros((8, 11), dtype=bool)
```

여기서는 NoC 링크가 방향별로 독립적인 대역폭을 가진다고 가정했다. 양방향 공유 대역폭을 사용하거나 라우터 내부에서 여러 포트가 대역폭을 공유한다면 그에 맞는 제약조건이 추가되어야 한다.

중요한 것은 커널의 대역폭 사용량을 `intra/outer`라는 두 개의 값으로만 저장하는 것이 아니라, 이를 실제 물리 링크의 대역폭 사용량으로 변환한다는 점이다.

\\[ L_k[\ell] = L\_{k,\text{intra}}[\ell] + L\_{k,\text{outer}}[\ell] \\]

`intra` 및 `outer`는 트래픽의 논리적인 분류이며, 물리 링크는 두 트래픽을 모두 처리한다.

### 3.2 Kernel Traffic Profiling

이를 위해 커널별로 트래픽 패턴을 모델링할 필요가 있다.

| 커널          | Intra-mesh                   | Outer-mesh             | DMA               |
| ----------- | ---------------------------- | ---------------------- | ----------------- |
| Linear      | 연산 타일 간 activation/weight 전달 | 외부 가중치 및 activation 로드 | 선택된 DMA 채널        |
| Reduction   | Partial sum 이동 및 집계          | 최종 결과 출력 등에 의해 결정      | 대체로 상대적으로 낮을 수 있음 |
| Elementwise | Local cache 활용 시 낮음          | 입력/출력 residency에 따라 다름 | 외부 데이터 전송 시 사용    |

이때 같은 `linear` 커널이라도 `core mesh`의 크기가 달라지면 트래픽 패턴이 달라질 수 있다.

예를 들어 \\(2\times 4\\)와 \\(4\times 4\\)의 `core mesh`에 동일한 커널을 할당한다면 연산 병렬성과 데이터 재사용이 달라지고, 그 결과 내부 NoC 트래픽, DMA 트래픽 및 실행 시간 모두 달라질 수 있다.

따라서 각 커널에 대해 여러 실행 구성(Variant)을 준비하는 방식을 추천한다.

```
from dataclasses import dataclass@dataclassclass Variant:    shape: tuple[int, int]    latency_ms: float    internal: list    external: list
```

`internal`은 `[(src, dst, GBps), ...]`로, `external`은 `[(dma_id, local_tile, GBps, is_read), ...]`로 표현할 수 있다.

여기서 `latency_ms`는 해당 `core mesh` 구성에서 단독 실행한 경우의 예상 latency이다.

### 3.3 Candidate Placement 탐색

각 커널에 대해 가능한 배치를 순회하면서 NoC 및 DMA 대역폭 제약조건을 검사한다.

Greedy Scheduler 동작 과정

Ready Kernel Queue

실행 가능한 커널 및 실행 구성

Candidate Generation

Core mesh 크기 및 배치 좌표 생성

Traffic Projection

XY Routing을 적용하여 링크별 트래픽 계산

Feasibility Check

Tile 점유 / NoC 링크 / DMA 채널 검사

Best Candidate Selection

예상 성능과 Resource Contention을 고려한 선택

대역폭 제약조건을 만족하는 후보들 중 최적의 후보를 선택해야 한다.

이를 위해 다음과 같은 Score를 사용할 수 있다.

\\[ Score(k,P)= \frac{ Priority_k/T_k(P) }{ 1+\alpha U\_{\max}(P) +\beta F\_{\text{tile}}(P) } \\]

여기서 각 항목은 다음을 의미한다.

- \\(Priority_k\\): 커널 실행 우선순위
- \\(T_k(P)\\): 해당 배치에서의 예상 실행 시간
- \\(U\_{\max}(P)\\): 배치 이후 가장 높은 NoC/DMA 자원 사용률
- \\(F\_{\text{tile}}(P)\\): 연산 타일 점유 비용 또는 fragmentation penalty

이 식은 최적성이 보장되는 목적함수가 아니라, Greedy 선택을 위한 초기 휴리스틱이다.

예상 latency를 줄이는 것과 공유 자원 사용률을 낮추는 것을 동시에 고려한다는 점에 의미가 있다.

### 3.4 Python + NumPy 프로토타입

위의 알고리즘을 구현한 간단한 실행 가능한 예시를 작성했다.

mta_spatial_scheduler_demo.py

Python · NumPy · 약 120줄

XY Routing, Link/DMA BW 제약 검사, Kernel Variant 및 Greedy Placement 구현

파일 열기&#x20;

Python 파일 다운로드

코드의 핵심적인 부분은 다음과 같다.

```
# Candidate의 예상 NoC 및 DMA 트래픽 계산dl, dd = self.demand(v, y, x)# 현재 자원 사용량에 신규 트래픽 추가nl = self.links + dlnd = self.dma + dd# 대역폭 상한의 90% 이내인 경우만 허용if (nl > 0.9 * self.link_cap).any():    continueif (nd > 0.9 * self.dma_cap).any():    continue# 병목 링크/채널의 utilization 계산bottleneck = max(    (nl / self.link_cap).max(),    (nd / self.dma_cap).max())
```

프로토타입에서는 8×11 Mesh와 2개의 DMA 채널을 정의하고 `linear`, `reduction`, `elementwise` 커널을 동시에 배치하는 테스트를 수행했다.

세 커널이 모두 배치되었으며, 테스트에 정의한 링크 및 DMA 상한 검사를 통과했다.&#x20;



다만 이는 임의로 설정한 트래픽 값을 사용하는 기능 검증용 예시이며, 실제 MTA 시뮬레이터와 비교한 성능 검증은 아니다.

## 4. 방법 3: MILP / CP-SAT 기반 최적화

세 번째 방법은 스케줄링 문제를 정수 최적화 문제로 공식화하는 것이다.

커널 \\(k\\)의 가능한 배치 후보를 \\(p\\)라 하고 다음 이진 변수를 정의하자.

\\[ x\_{k,p}= \begin{cases} 1 & \text{커널 }k\text{를 배치 }p\text{에 할당}\\\ 0 & \text{그렇지 않은 경우} \end{cases} \\]

이 변수를 바탕으로 커널별 배치 제한, Tile 점유 조건 및 NoC/DMA 대역폭 제한을 선형 제약조건으로 표현할 수 있다.

예를 들어 특정 NoC 링크 \\(\ell\\)에 대한 제약조건은 다음과 같다.

\\[ \sum_k\sum_p L\_{k,p}[\ell]x\_{k,p} \le B\_{\text{NoC}}[\ell] \\]

DMA 채널 \\(d\\) 역시 동일한 방식으로 표현한다.

\\[ \sum_k\sum_p D\_{k,p}[d]x\_{k,p} \le B\_{\text{DMA}}[d] \\]

그리고 다음 목적함수를 사용할 수 있다.

\\[ \max \sum_k\sum_p \frac{Priority_k}{T_k(p)} x\_{k,p} \\]

이는 선택된 커널들의 예상 처리율을 가중 합산한 근사 목적함수다.

이 접근 방법은 `scipy.optimize.milp`, OR-Tools CP-SAT 등을 활용해 구현할 수 있다.

다만 스케줄링 후보가 많아지면 계산량이 빠르게 증가한다. 따라서 실시간 Kernel Scheduler로 바로 사용하기보다는 Greedy 알고리즘의 최적성 차이를 측정하기 위한 Offline Oracle로 사용하는 것을 추천한다.

## 5. 실제 구현에서 추가로 중요한 세 가지 문제

### 5.1 대역폭은 평균값만으로 표현하기 어렵다

현재 제안한 방법에서는 커널별 트래픽 요구량을 고정된 GB/s 값으로 표현한다.

하지만 실제로는 DMA Prefetch, Compute, Writeback 등의 단계에 따라 NoC 대역폭 사용량이 변화한다.

예컨대 두 커널의 평균 DMA 대역폭 요구량이 각각 40GB/s여도 특정 구간에서 동시에 100GB/s를 요구할 수 있다.

따라서 초기 구현은 평균 대역폭을 사용하되, 이후에는 커널별 Phase-Aware Bandwidth Profile을 도입하는 것이 좋다.

\\[ R_k(P,t)= \left( C_k(P),L_k(P,t),D_k(P,t) \right) \\]

이는 커널의 실행 구성 및 실행 단계별 자원 사용량을 의미한다. 다만 시간별 프로파일을 합산하려면 커널들의 시작 시각 및 실행 단계 정렬도 함께 예측해야 한다.

### 5.2 DMA 채널과 NoC 대역폭 사이의 관계

DMA 채널에서 데이터를 가져오는 경우 실제 병목은 여러 지점에서 발생할 수 있다.

예를 들어 DMA 엔진이 초당 100GB를 읽어올 수 있어도, DMA가 연결된 NoC Router의 Injection Bandwidth가 50GB/s라면 실효 전송률은 더 낮아질 수 있다.

또한 DMA 엔진이 여러 개여도 실제로 동일한 HBM 채널을 공유할 수 있다.

따라서 가능하다면 다음 리소스를 별도로 모델링해야 한다.

| 리소스                | 제약조건                   |
| ------------------ | ---------------------- |
| DMA Engine         | DMA 엔진별 최대 처리율         |
| DMA Injection Port | DMA와 NoC 사이의 주입/배출 대역폭 |
| NoC Link           | 각 링크의 방향별 대역폭          |
| Memory Channel     | 공유하는 외부 DRAM 채널의 대역폭   |

이 부분은 아키텍처에서 공유 구조가 어떻게 설계되어 있는지에 따라 달라진다.

### 5.3 Core Mesh 크기와 실행 성능의 관계

이 부분이 스케줄러 성능을 결정하는 매우 중요한 요소라고 생각한다.

현재 Greedy 모델에서는 `core mesh`가 커질수록 연산 성능이 좋아진다고 가정하는 대신, 각 Variant의 실행 latency를 별도로 지정했다.

이것이 적절한 접근이다. 커널을 더 많은 연산 타일에 분산한다고 해서 성능이 항상 증가하는 것은 아니기 때문이다.

특히 Reduction처럼 통신이 증가할 수 있는 커널이나, 데이터 병렬성이 제한적인 Linear 커널의 경우 타일을 늘리는 것이 오히려 비효율적일 수 있다.

따라서 시뮬레이터에서 커널별로 다음의 Performance Profile을 미리 수집하는 방식이 유용하다.

\\[ Profile_k(h,w)= \left( T_k,\\, L\_{k,\text{intra}},\\, L\_{k,\text{outer}},\\, D_k \right) \\]

이때 단순한 총 대역폭보다는 타일 간 통신 패턴을 포함한 Link Traffic Profile을 저장하면 더 좋다.

또한 컴파일된 커널이 특정 Tile Grid에 고정되어 있다면 해당 커널을 임의 크기의 `core mesh`로 확대하거나 축소할 수는 없다. 실제로 지원되는 병렬화 및 매핑 구성만 Variant로 생성해야 한다.

## 6. 권장 구현 및 실험 순서

| 단계 | 구현 내용                          | 목적                |
| -- | ------------------------------ | ----------------- |
| 1  | Resource Vector Greedy         | 초기 Baseline       |
| 2  | Link-Aware Greedy + XY Routing | NoC contention 반영 |
| 3  | Mesh Size 및 Placement 공동 탐색    | 동적 타일 파티셔닝        |
| 4  | Phase-Aware Bandwidth Profile  | 시간적 대역폭 변동 반영     |
| 5  | MILP/CP-SAT Offline Oracle     | 최적성 Gap 평가        |
| 6  | Cycle-Level Simulator 비교       | 실제 성능 및 예측 정확도 검증 |

기존 문헌 중에서는 AuRORA (MICRO 2023)가 NUMA-aware compute placement와 동적 memory-resource partitioning을 함께 다루고 있다는 점에서 참고할 만하다.&#x20;

AuRORA.pdf



또한 vNPU (ISCA 2025)는 가상 NPU의 물리적 매핑과 NoC routing에 따른 간섭을 다루므로, Kernel Placement 설계와 관련성이 높다.&#x20;

vNPU.pdf



## 최종 제안

현재 구현하려는 스케줄러를 다음 두 계층으로 분리하는 것이 가장 좋겠다.

Global Kernel Scheduler

커널 우선순위 및 Co-location 조합 선정

Topology-Aware Placement Engine

Core Mesh 크기·위치 최적화 및 대역폭 제약 검사

MTA Cycle-Level Simulator

실행, NoC Contention, DMA Backpressure 측정

이렇게 분리하면 기존 FCFS, RR, FRFCFS 같은 Global Scheduling Policy를 유지하면서도 Placement Algorithm만 변경하여 실험할 수 있다.

또한 Placement 알고리즘의 성능은 단순한 타일 활용률뿐 아니라 전체 처리율, 평균/꼬리 latency, NoC 최대 링크 점유율, DMA stall, scheduling overhead로 비교하는 것이 적절하다.

결국 이 연구에서 흥미로운 핵심은 '어떤 커널들을 동시에 실행할 것인가'와 '각 커널을 어느 위치에, 몇 개의 타일로 배치할 것인가'를 함께 최적화하는 문제다. 이를 분리된 휴리스틱으로 시작해서 Joint Optimization으로 확장하면 연구적으로도 비교 가능한 설계 공간이 만들어질 것이다.