# Phase 10 — Bounded AFlow/ADAS Research Executor

## 구현 범위

Phase 10 실행기는 `cogni_flow.aflow_research`에 격리되어 있다. 기존
`WorkflowSpec`의 불변 tuple 구조와 DAG/노드/엣지/액션 예산을 재사용하되,
연구 실행기에서는 다음 규칙을 추가로 강제한다.

- 액션은 `Generate`, `Review`, `Revise`, `Ensemble`, `Test`, `Programmer`의
  닫힌 `WorkflowOperator` enum만 허용한다.
- 액션 인자는 연산자별 식별자 allowlist만 허용한다. 소스 코드, 셸 명령,
  실행 경로를 전달하거나 실행하는 표면은 없다.
- 모든 제안은 선택된 단일 부모와 동일한 계보 seed를 명시해야 한다. major
  변경도 다중 부모 결합을 사용할 수 없다.
- 후보는 held-in과 held-out 사례 각각을 최소 5회 반복 평가한다. 평균/표준
  편차, 지연, 자원, 도구 호출, 안전 위반을 함께 집계한다.
- 테스트 suite, 평가 policy, evaluator attestation, callable은 생성 시
  봉인된다. 교체나 policy/suite 변조가 감지되면 fail-closed 한다.
- held-out 비회귀, held-in/held-out 분산, 지연, 자원, 도구, 안전 게이트를
  모두 통과한 후보만 성공 trace가 된다.
- 성공/실패 trace는 각각 top-k로 제한되고, 반복 예산·archive 예산·개선
  patience에 의해 탐색이 종료된다.
- 승격 대상은 `research_archive_only` 하나뿐이다. 설치 함수, 프로덕션
  경로, 코드 덮어쓰기 callback은 존재하지 않는다.
- `IdleNightScheduler.for_research_workflow`와
  `ResearchWorkflowCoordinator`를 통해 추론이 멈춘 합법적인 evolution
  window에서만 연구 탐색을 예약할 수 있다.

## 검증

`tests/test_aflow_research.py`는 다음을 회귀 검증한다.

- 순환 DAG, 노드/엣지/액션/보관 예산, untyped 연산자와 코드 payload 거부
- 최소 5회 반복과 held-in/held-out 집계
- 동일 seed의 탐색 계보와 replay digest 재현
- evaluator/policy/result 변조 거부
- major 변경의 단일 부모 규칙
- 비회귀·분산·개선 gate
- top-k archive와 patience 조기 종료
- 기본 local Gemma proposer의 evidence gate
- 연구 archive 외 승격 경로 부재와 night scheduler 복귀

## 아직 주장하지 않는 증거

- 현재 테스트의 품질·지연 수치는 제어면 검증용 합성 evaluator 결과이며,
  Gemma 4 e4b의 실제 품질 또는 성능 증거가 아니다.
- 기본 local Gemma proposer는 실제 held-out 품질 보고서와 검증된 proposer
  artifact SHA-256이 공급되기 전까지 의도적으로 차단된다.
- `EvaluatorAttestation`은 배포 환경에서 실제 evaluator artifact의 SHA-256과
  연결해야 한다. 이 연결 전에는 연구 benchmark 결과를 제품 성능으로
  홍보할 수 없다.
- evaluator 자체의 sandbox 실행은 이 모듈의 책임이 아니다. Phase 10은
  실행 코드를 만들지 않고, 신뢰된 외부 evaluator가 반환한 bounded metric만
  검증·집계한다.
- replay digest는 입력·seed·evaluator가 동일하다는 감사를 돕지만, 비결정적
  GPU kernel까지 동일하게 만드는 증명은 아니다.
