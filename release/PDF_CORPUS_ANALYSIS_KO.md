# Cogni-OS 2.0 문헌 코퍼스 통합 분석

분석 대상: `C:\Project\Self-Harness\doc`의 PDF 10개, 총 270쪽  
분석일: 2026-07-10  
추출 결과: 10개 모두 텍스트 추출 성공, 페이지 추출 오류 0건

## 1. 핵심 결론

이 문헌군은 하나의 완성된 단일 시스템을 설명한다기보다, 다음 연구 사슬을
구성합니다.

1. **CTS**가 DEQ 고정점과 트리 탐색을 결합해 추론 깊이에 따른 활성 메모리
   증가를 억제합니다.
2. **System 1.5**가 수렴한 추론 궤적을 세션용 저랭크 Fast Weight로 압축해
   반복 질의의 시간 비용을 줄이려 합니다.
3. **System 2.5**가 FP-EWC와 스펙트럴 제약으로 장기 학습의 망각 및 고정점
   붕괴를 억제합니다.
4. **System 3**가 단일 가중치의 지식 포화 문제를 sparse implicit expert로
   분산합니다.
5. **System 4**가 가중치 업데이트 없이 사전 구성된 결합 토폴로지를 바꾸어
   스트리밍 분포 변화에 대응합니다.
6. **BIO-HAMA 특허**가 인지 상태 벡터와 전략·전술·반응 계층의 메타 라우팅을
   제시합니다.
7. **ADAS, AFlow, Self-Harness 및 자율 진화 개발 특허**가 위 인지 코어를
   설계·검증·개선하는 야간 제어면을 제시합니다.

가장 중요한 해석상의 제한은 다음과 같습니다.

- DEQ의 `O(1)`은 **고정된 상태 차원과 제한된 solver history에서 반복 깊이에
  대한 활성 메모리**입니다. 모델 가중치, 트리 통계, latent 저장소, expert bank,
  로그까지 포함한 전체 메모리가 상수라는 뜻은 아닙니다.
- 특허는 권리 범위와 실시 형태를 서술하며, 효과를 독립적으로 입증한 실험
  보고서가 아닙니다.
- `AAAI 27`, `ICLR 2026` 표기가 있는 문서는 로컬 원고 기준으로 분석했으며,
  게재·동료심사 상태는 별도로 확인하지 않았습니다.
- System 3 PDF에는 `[FILL]` 형태의 미완성 자리표시자가 있어, 수치 주장을
  구현 요구사항으로 곧바로 확정해서는 안 됩니다.

## 2. 문서별 분석

### 2.1 자율 진화형 소프트웨어 개발 시스템 — 47쪽

파일: `10-2025-0114865_허상훈_명세서 최종본.pdf`

- **목표:** 자연어 요구사항에서 지식 통합, 아키텍처 설계, 코드 생성, 검증,
  배포 및 운영 피드백까지 개발 생명주기를 자동화합니다.
- **핵심 구성:** Hybrid RAG형 지식 통합, 아키텍처 패턴 라이브러리,
  NSGA-II 다목적 최적화, proposer/solver Self-Play, 정적·동적·형식 검증,
  피드백 기반 재설계입니다.
- **청구의 중심:** 요구사항→통합 컨텍스트→최적 아키텍처→설계안→Self-Play
  코드→완성 소프트웨어의 연쇄와 격리 실행·다단계 검증입니다.
- **Cogni-OS 반영:** Cogni-Flow의 후보 워크플로 탐색, 회귀 검증, 원자적 승격,
  실패 로그 환류 구조의 상위 요구사항입니다.
- **주의점:** 외부 지식·API를 전제한 실시 형태는 100% air-gap 요구와 직접
  충돌합니다. 오프라인 미러, 서명된 artifact, 로컬 취약점 DB로 치환해야 합니다.

### 2.2 뇌과학 기반 동적 모듈형 AI 시스템 — 54쪽

파일: `10-2025-0141637_허상훈_특허출원서 (1).pdf`

- **목표:** 단일 거대 모델 대신 필요한 인지 모듈만 선택하고, 메타인지 상태로
  실행 경로를 바꾸는 인지 스캐폴딩 구조를 제안합니다.
- **인지 상태:** 작업 기억, 감정 맥락, 주의 자원, 예측 불확실성, 인지 부하의
  다중 특성 벡터입니다.
- **라우팅:** 전략층은 목표 표현, 전술층은 모듈 조합과 순서, 반응층은 오류·
  불확실성 상승에 따른 중단 또는 재계획을 담당합니다.
- **학습:** 성능·비용 보상으로 정책·가치 파라미터를 갱신하는 Bio-A-GRPO
  계열 메타 학습을 기술합니다.
- **Cogni-OS 반영:** BIO-HAMA 메타 라우터의 5요소 텐서 상태, 계층별 top-k
  mask, 동적 학습률·할인율의 설계 근거입니다.
- **주의점:** 감정·사회인지 같은 고수준 개념은 측정 가능한 tensor proxy와
  검증 지표로 다시 정의해야 재현 가능한 구현이 됩니다.

### 2.3 System 1.5: Post-Deliberation Fast Weights — 12쪽

- **문제:** DEQ는 메모리를 절약하지만 유사한 후속 질의에도 고정점 궤적을
  다시 풀어 시간 비용이 반복됩니다.
- **제안:** 수렴 상태 또는 추론 흔적을 저랭크 가중치 오버레이로 컴파일하고,
  품질·불확실성 gate와 Near-OOD router를 거쳐 세션 내에서만 사용합니다.
- **안전 장치:** base weight 불변, 저랭크·스펙트럴 제약, out-of-distribution
  입력의 원래 solver fallback, 세션 종료 시 제거입니다.
- **논문 주장:** 결론부는 평균 3.6배, 최선 8.4배의 amortized speedup을
  보고하지만, 이 수치는 현재 로컬 구현의 독립 재현 결과가 아닙니다.
- **Cogni-OS 반영:** 예외 안전 forward hook, CPU LRU session cache, operator
  norm 및 empirical quality admission, OOD fallback으로 구현했습니다.
- **미해결:** 실제 Gemma 규모에서 Fast Weight가 정답 품질을 보존·회복하는지는
  별도 학습과 장기 benchmark가 필요합니다.

### 2.4 System 2.5: O(1)-Memory Lifelong Learning / FP-EWC — 14쪽

- **문제:** DEQ를 순차 도메인에 학습하면 stability-plasticity 충돌과 파국적
  망각이 생기며, 일반적인 명시적 gradient/SVD 보존법은 메모리 이점을 훼손할
  수 있습니다.
- **제안:** 고정점에서 IFT와 matrix-free 계산으로 Fisher 정보를 추정하는
  FP-EWC, 중요 파라미터의 anchor penalty, C-FIRE 스펙트럴 투영을 결합합니다.
- **수학적 요구:** 고정점 map의 수축성과 선형계 solver의 수렴이 전제입니다.
  `L < 1`을 선언만 해서는 부족하며 매 update 전후 투영 또는 hard rejection이
  필요합니다.
- **Cogni-OS 반영:** dense Jacobian을 만들지 않는 Fisher-vector 경로, 제한된
  domain snapshot, 오래된 quadratic merge, optimizer 전후 projection입니다.
- **미해결:** toy/domain 단위 검증과 실제 장기 망각 방지 benchmark는 서로 다른
  증거 수준입니다.

### 2.5 System 3: Sparse Implicit Mixtures — 18쪽

- **문제:** 하나의 공유 가중치가 계속 새로운 도메인을 흡수하면 rank saturation과
  간섭으로 이른바 Capacity Wall에 도달한다는 가설입니다.
- **제안:** 상태 `z`에 의존하지 않는 sparse router, contractive gated mixture,
  novelty 기반 expert recruitment, expert별 FP-EWC를 결합합니다.
- **중요 제약:** expert를 무제한 생성하면 24GB 목표와 모순됩니다. 실제 구현은
  사전 할당한 고정 pool, top-k 실행, usage 기반 재활용·merge가 필요합니다.
- **Cogni-OS 반영:** 고정 최대 expert 수, 결정적 novelty recruitment, parameter
  및 VRAM admission, sparse top-k, spectral projection을 구현했습니다.
- **주의점:** PDF의 일부 표·수치에 `[FILL]` 자리표시자가 있어 실험 결과를
  확정된 근거로 취급하지 않았습니다.

### 2.6 AFlow: Workflow Optimization — 38쪽

- **목표:** 사람이 초기 workflow를 완전히 설계하는 대신, node/edge/action으로
  표현된 agent workflow를 반복 탐색·평가합니다.
- **핵심:** workflow 최적화를 MDP/탐색 문제로 보고, archive의 성능 좋은 후보와
  탐색 다양성을 함께 이용합니다.
- **Cogni-OS 반영:** immutable DAG spec, node/edge/action hard budget, 안정화한
  soft-mixed selection, 제한된 반복·archive, evaluator 의존 평가로 구현했습니다.
- **안전 해석:** 생성된 문자열을 곧바로 `eval/exec`하는 방식은 사용하지 않아야
  합니다. 후보는 inert data로 유지하고 별도 격리 검증 경계를 통과해야 합니다.

### 2.7 Automated Design of Agentic Systems (ADAS) — 34쪽

- **목표:** prompt만 최적화하는 범위를 넘어, agent의 building block과 조합
  자체를 메타 에이전트가 코드로 발명·탐색하는 연구 방향을 정의합니다.
- **방법:** Meta Agent Search가 이전 archive를 바탕으로 새로운 agent program을
  제안하고 benchmark 점수로 반복 선택합니다.
- **장점:** 모듈·제어 흐름·역할 분담을 함께 탐색할 수 있습니다.
- **위험:** benchmark overfitting, 자기평가 편향, 임의 코드 실행, 탐색 비용이
  핵심 리스크입니다.
- **Cogni-OS 반영:** 제안·평가 권한을 분리하고, 정적 budget과 회귀 gate를 먼저
  통과한 후보만 야간 진화 단계에서 다룹니다.

### 2.8 Cognitive Tree Search (CTS) — 21쪽

- **문제:** autoregressive MCTS가 각 prefix의 KV cache를 저장하면 깊이·분기에
  따라 GPU 메모리가 증가합니다.
- **제안:** 각 node transition을 DEQ fixed-point solve로 바꾸고, search 중에는
  텍스트/KV cache 대신 latent state와 제한된 semantic ancestor 검색을 사용합니다.
- **필수 조건:** solver history 고정, cache mutation 금지, 수렴 실패 hard-stop 또는
  명시적 fallback, 트리 메타데이터의 별도 고정 capacity입니다.
- **Cogni-OS 반영:** limited-history multisecant solver, cache-free Gemma decoder
  adapter, 고정 capacity PUCT/latent retrieval, depth-memory gate입니다.
- **정확한 O(1) 해석:** solver activation은 depth-independent로 만들 수 있지만,
  모든 과거 tree node를 보존하면 tree memory는 여전히 증가합니다. 고정 arena와
  eviction/rollout 정책을 써야 전체 search working set이 bounded가 됩니다.

### 2.9 Self-Harness — 19쪽

- **목표:** 더 강한 외부 agent나 인간 엔지니어 없이, base agent가 자신의 harness를
  실패 데이터와 검증 결과에 따라 개선합니다.
- **구성:** 실패 수집, weakness clustering, 후보 harness 생성, sandbox evaluation,
  accept/reject 및 반복입니다.
- **Cogni-OS 반영:** 로컬 실패 DB, verifier signature 기반 clustering, 변경 파일
  allowlist, base digest, staging tree, regression test, atomic promotion, audit log입니다.
- **절대 안전 조건:** subprocess와 AST blocklist만으로는 악성·오류 코드를 격리할
  수 없습니다. 자동 실행·승격은 네트워크와 호스트 파일시스템을 커널 수준에서
  차단하는 VM/container runner가 있을 때만 허용해야 합니다.

### 2.10 System 4: Gradient-Free Topology Switching TTA — 13쪽

- **문제:** 기존 test-time adaptation은 gradient update 지연, 연속 분포 변화에서의
  망각, 안정성 보장 부족을 가집니다.
- **제안:** coupled equilibrium micro-agent의 가중치를 바꾸지 않고, 사전 계산한
  결합 토폴로지를 PCAS류 change score로 선택합니다.
- **안전 조건:** 각 local operator와 coupling의 합성 Lipschitz bound가 1 미만이어야
  하며, topology 후보 자체도 사전 검증되어야 합니다.
- **Cogni-OS 반영:** 고정된 agent 수, 사전 구성 DAG topology, Mahalanobis change
  score, tensor-only topology mask, 제한된 warm iteration을 구현했습니다.
- **미해결:** sub-millisecond 및 스트리밍 정확도 주장은 대상 GPU·입력률에서 별도
  latency/quality benchmark가 필요합니다.

## 3. 통합 아키텍처로 번역한 요구사항

| 문헌 축 | 실행 요구사항 | 실패 시 정책 |
|---|---|---|
| CTS/DEQ | cache-free transition, 제한 history, 고정 search arena | 미수렴 상태를 성공으로 표시하지 않고 reject/fallback |
| System 1.5 | base 불변 저랭크 overlay, quality/OOD gate | overlay 제거 후 원래 solver 경로 |
| System 2.5 | matrix-free Fisher, update 전후 spectral projection | update 거부·rollback |
| System 3 | sparse top-k, 고정 expert pool, bounded recruitment | recycle/merge 또는 spawn 거부 |
| System 4 | tensor change score, 사전 검증 topology | 안정 기본 topology |
| BIO-HAMA | 5요소 인지 tensor, 계층적 budget mask | 가용 module만 선택 |
| AFlow/ADAS | immutable workflow, 탐색·archive hard cap | 후보 폐기 |
| Self-Harness | day/night 배타, kernel sandbox, 회귀검증, 원자승격 | 원본 보존·safe mode |

## 4. 구현 우선순위

1. 수렴·IFT·비수축 실패의 세 수학 gate를 먼저 고정합니다.
2. 실제 로컬 Gemma 한 레이어에 DEQ를 주입하고 finite output·잔차·VRAM을
   측정합니다.
3. 고정 메모리 CTS를 연결합니다.
4. Fast Weight는 실제 품질 gate를 통과한 세션에서만 활성화합니다.
5. FP-EWC/C-FIRE를 모든 학습·restore 경로에 강제합니다.
6. expert/swarm/meta-router를 고정 budget으로 추가합니다.
7. 마지막에만 AFlow/Self-Harness를 연결하며, 커널 격리가 없으면 후보 코드를
   실행하지 않습니다.

## 5. 최종 판정

문헌들은 서로 보완적인 연구 방향을 제시하지만, 문서의 조합만으로 “AGI” 또는
“무한 진화”가 입증되지는 않습니다. 실제 시스템의 합격 기준은 명칭이 아니라
다음 관측값이어야 합니다: 고정점 잔차, spectral bound, 실제 peak VRAM, depth별
allocation 변화, 품질 회귀, 망각률, OOD fallback 정확도, sandbox 탈출 테스트,
원자적 rollback 성공률. Cogni-OS 구현은 이 관측 가능한 gate를 중심으로 구성해야
합니다.
