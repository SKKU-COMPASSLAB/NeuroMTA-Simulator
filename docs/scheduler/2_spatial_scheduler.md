
커널 후보를 파악하고, 우선순위에 따라 커널들을 정렬한다. (scheduling policy에 따른 것으로 FCFS, FRFCFS, RR 등을 현재 지원하고 있음)

kernel window K 개의 상위 후보들에 대해 co-location할 지 여부를 분석하고, 동시에 실행할 커널 N개를 선택한다. 선택의 기준은 roofline model (NoC 제외)에 따라 현재 점유된 리소스를 제외한 나머지 리소스에 대해 N개의 커널을 실행할 때 bottleneck이 발생하지 않는지 여부이다.

커널의 연산 특성 및 데이터 재사용 패턴에 따라 intra-mesh/outer-mesh traffic를 예측하고, 이를 기반으로 DAG를 생성한다. DAG의 각 노드는 논리 연산기 코어 그룹 (logical core mesh)과 DMA 채널로 이루어지며, 이들을 연결하는 링크의 weight은 outer-mesh traffic을 의미한다.&#x20;

논리 연산기 코어 그룹을 실제로 placement한다. 이 과정에서 각 logical core mesh의 형태가 정해지며, 그 위치 또한 정해진다. 다만, 각 core mesh의 형태 및 위치는 traffic이 고르게 분포하도록 해야 한다. 모든 NoC 링크에 대해 편차가 제일 작은 후보를 선택하거나, greedy하게 최적의 솔루션을 도출한다.

$$s=\alpha \times \log_2{\beta}$$