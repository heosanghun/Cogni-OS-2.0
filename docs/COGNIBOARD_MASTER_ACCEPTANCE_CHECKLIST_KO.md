# CogniBoard 마스터 완료 기준 체크리스트 (ID 1–170)

## 0. 문서 목적과 판정 기준

이 문서는 사용자가 요청한 기능을 빠짐없이 고정 ID로 추적하는 단일 완료 원장이다. 판정 기준은
현재 v0.4.1 작업 소스이다. 항목별 상태는 현재 코드 경로와 동일 exact commit·model·config·
device 범위의 자동 검증 또는 실측 증거만 반영한다. v0.4.0의 EXE·wheel·source ZIP·manual·
SBOM·checksum과 실행 smoke는 historical comparison이며 현재 소스의 완료 증거가 아니다.
현재 exact commit의 clean CPU gate, GPU5 guard 증거, 독립 verifier와 새 release bytes가
결합되기 전에는 과거 binary·JSON·GitHub SHA를 승격 근거로 재사용하지 않는다.

상태의 의미는 다음과 같다.

| 상태 | 의미 |
|---|---|
| `COMPLETED` | 현재 소스에 실행 경로가 연결되어 있고, 승인된 독립 verifier가 서명한 exact-scope EvidenceRecordV1·raw artifact·raw payload가 해당 acceptance ID를 명시적으로 승인한다. |
| `IMPLEMENTED_UNVERIFIED` | 코드·설정·정적 artifact 경로는 있으나 현재 exact clean CPU/static attestation과 승인된 서명이 아직 결합되지 않았다. 완료로 계산하지 않는다. |
| `PARTIAL` | 일부 경로는 있으나 제품 연결, 독립 검증, 현재 소스 회귀 또는 요구 범위 일부가 남았다. |
| `NOT_IMPLEMENTED` | 제품에서 실행 가능한 경로가 없거나 의도적으로 차단되어 있다. |
| `EXTERNAL_BLOCKER` | 코드만으로 승격할 수 없으며 지정 하드웨어·승인 토큰·라이선스·외부 독립 평가가 필요하다. |

`[x]`는 위 정의의 `COMPLETED`만 뜻한다. `PARTIAL`이나 외부 자료가 없는 상태를 화면 표시,
설계 문서, 단위 테스트만으로 완료라고 부르지 않는다.

현재 스냅샷 집계는 `COMPLETED 0 / IMPLEMENTED_UNVERIFIED 98 / PARTIAL 59 / NOT_IMPLEMENTED 8 /
EXTERNAL_BLOCKER 5`이다. 즉 170개 모두 아직 승인된 완료 증거 승격 조건이 남아 있다. 과거 20-turn
대화·CTS·패키지 관측은 current-scope 증거가 아니며, clean exact-commit CPU gate와
GPU5 guard 측정, 독립 검증, 새 패키지 smoke 전에는 승격하지 않는다.

### 0.1 `COMPLETED` 승격 증거 계약

`COMPLETED` 행의 실제 근거 열에는 다음 두 토큰이 각각 정확히 하나 있어야 한다.

```text
`basis=STATIC_ARTIFACT|CPU_VERIFIED|MODEL_MEASURED|GPU_MEASURED|EXTERNAL_VERIFIED`
(release|validation)/evidence/<record>.json#sha256=<64자리 소문자 SHA-256>
```

한 EvidenceRecordV1이 여러 acceptance ID를 함께 승인할 수는 있지만, 서명된 `claim_ids`에 각
`acceptance.id<N>`이 명시되어야 한다. record의 canonical `evidence_id`, exact bytes citation,
raw artifact·payload digest, source/model/config/device scope, verifier ID와 공개 키, 분리된 signature가
모두 일치해야 한다. 공개 키는 고정된 `config/release-verifier-policy.json`에서 `approved`로 승인된
verifier와 정확히 일치해야 하며, 정책이 `unconfigured`이면 모든 `COMPLETED` 승격은 fail-closed다.
운영자가 CLI로 입력한 digest, 임의 prose, 과거 JSON 경로, 체크박스 변경만으로는 승격할 수 없다.

각 ID의 허용 basis/kind, 필수 component와 raw artifact schema는
`config/acceptance-evidence-policy.json`에 1–170 전부 명시되어 있고 validator 소스가 그 exact-byte
SHA-256을 고정한다. 따라서 일반 정적 파일 하나로 GPU 실측, 실제 모델 대화, Lens 공식 응답 또는
GitHub 공개 상태를 대신 승인할 수 없다. record·raw payload·raw artifact는 이 정책 SHA와 동일한
component 집합을 결합해야 한다. 완료 원장과 증거는 immutable release-candidate SUBJECT tree 밖의
detached bundle에 둔다. validator와 renderer는 사용자가 입력한 commit/digest 문자열을 신뢰하지
않고 source-pinned verifier key로 서명된 기존 release attestation에서 SUBJECT commit·source
tree·model·config·device scope를 파생한다. `COMPLETED`가 있는 유효 원장은 이 detached attestation이
없거나 어느 scope라도 서명 record와 다르면 fail-closed다.

## 1. 전체 기능 체크리스트

### A. 기본 런타임과 제품 경계 (1–9)

| ID | 체크 | 요구사항 | 상태 | 실제 근거 | 완료 승격 조건 |
|---:|:---:|---|---|---|---|
| 1 | [ ] | Gemma 4 E4B-it 로컬 백본 | `IMPLEMENTED_UNVERIFIED` | `cogni_agent/model_service.py`, `config/gemma4-e4b-it.manifest.toml`, `tests/test_agent_model_service.py` | 현재 동결 아티팩트·manifest 계약 유지 |
| 2 | [ ] | effective 파라미터 약 4.5B 표시 | `IMPLEMENTED_UNVERIFIED` | `cogni_os/artifacts.py`, `cogni_os/factbook.py`, `tests/test_artifacts.py` | 모델 교체 시 실제 텐서 inventory 재계산 |
| 3 | [ ] | 저장 파라미터와 effective 파라미터 구분 | `IMPLEMENTED_UNVERIFIED` | `cogni_os/artifacts.py`, `tests/test_agent_fact_grounding.py` | Fact-book 수치와 manifest 일치 유지 |
| 4 | [ ] | RTX 4090 24GB 목표 장치 검증 | `EXTERNAL_BLOCKER` | 목표는 `config/default.toml`; 기존 실측은 `release/evidence/phase11_gemma_cts_runtime_20260712.json`의 RTX 5090 Laptop | 정확한 RTX 4090에서 동결 소스·E4B-it 전체 게이트 재실측 |
| 5 | [ ] | 실제 장치와 목표 장치 구분 표시 | `IMPLEMENTED_UNVERIFIED` | `cogni_os/factbook.py`, `cogni_demo/static/app.js`, `tests/test_factbook.py` | 라이브 검증 전에는 `NOT VERIFIED` 유지 |
| 6 | [ ] | 기본 air-gap 및 외부 호출 0 | `PARTIAL` | `cogni_os/runtime.py`, `cogni_demo/server.py`, `tests/test_airgap.py` | 현재 배포 바이트에 대한 패킷/egress 감사와 오프라인 의존성 검토 |
| 7 | [ ] | 외부 API는 명시적 온라인 모드에서만 허용 | `IMPLEMENTED_UNVERIFIED` | `cogni_demo/workspace_capabilities.py::WebAccessPolicy`, `tests/test_workspace_capabilities.py` | 모든 향후 connector가 동일 권한 계약 사용 |
| 8 | [ ] | 최신 기능 통합 더블클릭 EXE | `PARTIAL` | v0.4.0 EXE·cold-start smoke는 historical; v0.4.1 빌더와 무서명 상태 표시는 구현됨 | current exact commit에서 evidence-bound EXE 재생성·checksum·cold-start smoke |
| 9 | [ ] | 패키지된 EXE 기능 회귀 | `PARTIAL` | v0.4.0 bundle JSON은 historical; 현재 release 회귀 계약은 tests에 존재 | v0.4.1 clean bundle의 인증 UI/state/capability/shutdown과 현재 모델 smoke |

### B. Cogni-Core: DEQ·CTS (10–18)

| ID | 체크 | 요구사항 | 상태 | 실제 근거 | 완료 승격 조건 |
|---:|:---:|---|---|---|---|
| 10 | [ ] | Gemma 특징과 DEQ 고정점 경로 결합 | `PARTIAL` | `cogni_agent/core_pipeline.py`, `cogni_core/deq.py`, `tests/test_core_pipeline.py` | 학습된 DEQ/Wproj와 held-out 인과 ablation으로 canary 해제 |
| 11 | [ ] | Broyden/L-Broyden bounded solver | `IMPLEMENTED_UNVERIFIED` | `cogni_core/deq.py`, `cogni_core/search.py`, `tests/test_deq.py`, `tests/test_search.py` | rank/history 상한 회귀 유지 |
| 12 | [ ] | CTS Depth 100 이상 | `IMPLEMENTED_UNVERIFIED` | `cogni_core/cts.py`, `scripts/validate_gemma4_runtime.py`, Phase 11 runtime JSON | 같은 범위에서 100/100·finite 유지 |
| 13 | [ ] | 고정 301-node arena | `IMPLEMENTED_UNVERIFIED` | `cogni_core/cts.py`, `tests/test_cts_policy.py` | arena 초과 fail-closed 유지 |
| 14 | [ ] | 추론 깊이에 선형 증가하지 않는 메모리 | `PARTIAL` | CTS arena와 solver history는 고정; `docs/VALIDATION.md` | 모델·로그·외부 데이터 전체 O(1) 주장은 금지하고 심도별 실제 곡선 재측정 |
| 15 | [ ] | KV-cache 비축 금지 | `IMPLEMENTED_UNVERIFIED` | `cogni_agent/model_service.py`의 `use_cache=False`, `tests/test_agent_model_service.py` | 모든 decode 경로에서 cache 비활성 유지 |
| 16 | [ ] | 16.7 GiB VRAM 방어선 | `EXTERNAL_BLOCKER` | `scripts/validate_gemma4_runtime.py` postcondition; 과거 5090 실측 14.8469 GiB | 목표 RTX 4090·E4B-it·현재 소스에서 allocated/reserved 모두 재검증 |
| 17 | [ ] | residual·finite·fallback 명시 | `IMPLEMENTED_UNVERIFIED` | `cogni_core/deq.py`, `cogni_os/evidence.py`, runtime validator/tests | 실패 시 답변 미게시 계약 유지 |
| 18 | [ ] | 실제 GPU 메모리/장치 프로파일링 | `PARTIAL` | `scripts/validate_gemma4_runtime.py` 경로는 구현; 과거 Phase 11 JSON은 current scope 아님 | current exact commit·E4B-it·GPU5 guard에서 allocated/reserved/device 증거 |

### C. System 1.5·2.5·3·4 (19–30)

| ID | 체크 | 요구사항 | 상태 | 실제 근거 | 완료 승격 조건 |
|---:|:---:|---|---|---|---|
| 19 | [ ] | System 1.5 Fast Weight | `PARTIAL` | `cogni_core/fast_weights.py`, `tests/test_fastweight_runtime.py`; Fact-book `gated` | 승인된 학습 checkpoint와 AQ/OOD·비회귀 증거 |
| 20 | [ ] | DEQ 상태를 저랭크 임시 가중치로 변환 | `PARTIAL` | `cogni_core/fast_weights.py`, `cogni_core/adaptation.py` | 실제 학습 artifact로 answer-quality 이득 검증 |
| 21 | [ ] | 검증 세션에서만 Fast Weight 활성 | `IMPLEMENTED_UNVERIFIED` | `cogni_core/fast_weight_safety.py`, `tests/test_fast_weight_safety.py` | TTL/LRU/session/artifact binding 유지 |
| 22 | [ ] | System 2.5 FP-EWC | `PARTIAL` | `cogni_core/fp_ewc.py`, `tests/test_fp_ewc.py`; `night_only` | 실제 도메인 3-seed BWT/FWT 증거 |
| 23 | [ ] | Fisher 기반 망각 방지 | `PARTIAL` | empirical Fisher/anchor/merge 구현과 `tests/test_fp_ewc.py` | 독립 데이터셋의 망각 감소와 품질 비회귀 |
| 24 | [ ] | C-FIRE spectral norm < 0.95 | `PARTIAL` | `cogni_core/c_fire.py`, `tests/test_c_fire.py` | 전체 적용 대상과 합성 연산의 인증서·실측; 전체 decoder 수축 주장 금지 |
| 25 | [ ] | System 3 Sparse MoE | `PARTIAL` | `cogni_core/experts.py`, `cogni_core/expert_lifecycle.py`, Phase 8 tests | 학습·교정 expert checkpoint와 독립 verifier |
| 26 | [ ] | novelty 기반 routing | `PARTIAL` | `cogni_core/routing.py`, `tests/test_routing.py` | 실제 OOD corpus threshold calibration |
| 27 | [ ] | 안전한 expert spawn·등록·rollback | `PARTIAL` | `cogni_core/expert_lifecycle.py`, `tests/test_expert_lifecycle.py` | 제품 artifact promotion의 독립 검증 |
| 28 | [ ] | System 4 Tensor Swarm | `PARTIAL` | `cogni_core/swarm.py`, `cogni_core/swarm_sessions.py`, `tests/test_swarm.py`; `advisory` | production PCAS corpus와 답변 품질 ablation |
| 29 | [ ] | 내부 에이전트 tensor 중심 통신 | `PARTIAL` | System 4 tensor 경로와 `cogni_os/tensor_service.py`; 제어/제품 전체가 tensor-only는 아님 | 모든 대상 경계의 명시적 tensor schema·복사/직렬화 측정 |
| 30 | [ ] | System 4 latency 검증 | `PARTIAL` | CPU benchmark 경로는 구현; 과거 System 4 CUDA JSON은 historical | current exact commit의 GPU5 guard에서 p50/p95/p99·수렴·finite 재측정 |

### D. Cogni-Flow·오케스트레이션 (31–37)

| ID | 체크 | 요구사항 | 상태 | 실제 근거 | 완료 승격 조건 |
|---:|:---:|---|---|---|---|
| 31 | [ ] | BIO-HAMA meta-router | `PARTIAL` | `cogni_core/meta_router.py`, `tests/test_meta_router.py`; `advisory` | 실제 routing quality calibration |
| 32 | [ ] | ADAS/AFlow pipeline 생성·탐색 | `PARTIAL` | `cogni_flow/aflow.py`, `cogni_flow/aflow_research.py`; `research_archive_only` | attested evaluator와 제품 설치 권한 분리 검증 |
| 33 | [ ] | 계획→실행→검토→재계획 루프 | `IMPLEMENTED_UNVERIFIED` | `cogni_flow/orchestrator.py`, `cogni_flow/task_plan.py`, 관련 tests | bounded 횟수·deadline 유지 |
| 34 | [ ] | bounded MCTS/workflow search | `IMPLEMENTED_UNVERIFIED` | `cogni_flow/aflow.py`, `tests/test_aflow.py` | archive/후보/노드 상한 유지 |
| 35 | [ ] | CPU control plane / GPU data plane 분리 | `IMPLEMENTED_UNVERIFIED` | `docs/ARCHITECTURE.md`, `cogni_os/runtime.py` | 단일 GPU owner 계약 유지 |
| 36 | [ ] | 단일 GPU lease owner | `IMPLEMENTED_UNVERIFIED` | `cogni_os/gpu_lease.py`, `tests/test_gpu_lease.py` | epoch/job/deadline binding 유지 |
| 37 | [ ] | 주간 inference/야간 evolution 상호 배제 | `IMPLEMENTED_UNVERIFIED` | `cogni_flow/rhythm.py`, `tests/test_rhythm.py` | 동시 진입 fail-closed 유지 |

### E. 자연스러운 대화 품질 (38–52)

| ID | 체크 | 요구사항 | 상태 | 실제 근거 | 완료 승격 조건 |
|---:|:---:|---|---|---|---|
| 38 | [ ] | 하드코딩이 아닌 로컬 Gemma 생성 | `IMPLEMENTED_UNVERIFIED` | `cogni_agent/model_service.py`, `cogni_agent/manager.py`, actual-model validation JSON | 모델 실패를 템플릿 정상 답변으로 위장하지 않기 |
| 39 | [ ] | 임의의 일반 질문 응답 | `PARTIAL` | 실제 Gemma 대화 경로 존재; typed tool 권한은 제한 | 독립 human-labelled 다양한 주제 acceptance |
| 40 | [ ] | 자연스러운 한국어 대화 | `PARTIAL` | validator와 품질 계약은 구현; v0.4.0 10/10 JSON은 historical | current exact commit·E4B-it GPU5 guard에서 10-turn strict gate |
| 41 | [ ] | bounded 다중 turn 문맥 | `IMPLEMENTED_UNVERIFIED` | `cogni_agent/conversation.py`, `tests/test_agent_conversation.py` | 압축/제거 시 turn 무결성 유지 |
| 42 | [ ] | 주제 전환 처리 | `IMPLEMENTED_UNVERIFIED` | conversation/quality integration tests | 주제어 억지 복사 없이 회귀 유지 |
| 43 | [ ] | 동일 문장 무한 반복 방지 | `IMPLEMENTED_UNVERIFIED` | `cogni_agent/response_quality.py`, completion stress evidence | 최신 소스 실모델 stress 재실행 |
| 44 | [ ] | 답변 중간 끊김 방지 | `IMPLEMENTED_UNVERIFIED` | `cogni_agent/manager.py`, `tests/test_agent_completion_stress.py` | terminal stop/timeout/final sentence gate 유지 |
| 45 | [ ] | 완결 문장 게시 | `IMPLEMENTED_UNVERIFIED` | response quality tests | 형식 요청과 자연문장을 구분해 strict gate 유지 |
| 46 | [ ] | 길이 한계 후 이어쓰기 | `IMPLEMENTED_UNVERIFIED` | `cogni_agent/conversation.py`, manager continuation tests | 중복 prefix 없는 동일 turn 연속성 유지 |
| 47 | [ ] | USER/ASSISTANT/control token 누출 방지 | `IMPLEMENTED_UNVERIFIED` | `cogni_agent/manager.py`, `tests/test_response_quality.py` | 신규 tokenizer special token 회귀 추가 |
| 48 | [ ] | 사용자 질문 echo 방지 | `IMPLEMENTED_UNVERIFIED` | manager echo stripping과 `tests/test_agent_manager.py` | 인용 요청은 오탐 제거하지 않기 |
| 49 | [ ] | 모델 정체성·파라미터 hallucination 방지 | `IMPLEMENTED_UNVERIFIED` | `cogni_agent/fact_grounding.py`, `tests/test_agent_fact_grounding.py` | Fact-book 외 수치 생성 차단 유지 |
| 50 | [ ] | 아키텍처·상태를 실제 capability로 설명 | `IMPLEMENTED_UNVERIFIED` | `cogni_os/capabilities.py`, Fact-book, grounding tests | `gated/advisory`를 active로 표현하지 않기 |
| 51 | [ ] | 질문당 assistant 답변 정확히 1개 | `IMPLEMENTED_UNVERIFIED` | transactional manager/conversation tests, validation addendum | 재시도 후보를 UI에 중복 게시하지 않기 |
| 52 | [ ] | 현재 소스 actual-model 품질 stress | `PARTIAL` | v0.4.0 10/10·20/20 JSON은 historical; stress validator와 회귀 tests는 구현 | current exact commit·E4B-it GPU5 guard에서 동일 corpus strict gate |

### F. 로컬 작업 에이전트 (53–64)

| ID | 체크 | 요구사항 | 상태 | 실제 근거 | 완료 승격 조건 |
|---:|:---:|---|---|---|---|
| 53 | [ ] | 프로젝트 파일 읽기·검색 | `IMPLEMENTED_UNVERIFIED` | `cogni_agent/tools.py`, `tests/test_agent_tools.py` | root/path/reparse/size gate 유지 |
| 54 | [ ] | 코드 작성·수정 | `PARTIAL` | output-only write와 T2 proposal staging; 실제 source mutation 금지 | 승인·격리·회귀·rollback이 결합된 source patch 경로 |
| 55 | [ ] | PoC/MVP 개발 지원 | `PARTIAL` | `cogni_agent/tools.py`의 bounded `/project` 다중 파일 묶음, `cogni_flow/task_plan.py`, `tests/test_agent_project_bundle.py`; output-only·무실행·무덮어쓰기 | 자연어 요청→검토 가능한 typed bundle 생성과 선택적 격리 테스트 E2E |
| 56 | [ ] | 테스트 실행 | `PARTIAL` | 고정 pytest primitive, `cogni_flow/task_plan.py`; 기본 차단 | OS 격리와 명시적 trusted opt-in |
| 57 | [ ] | 결과·실패 로그 저장 | `IMPLEMENTED_UNVERIFIED` | `cogni_flow/logdb.py`, `tests/test_logdb.py` | 민감정보/크기/보존정책 유지 |
| 58 | [ ] | 계획·진행률 표시 | `IMPLEMENTED_UNVERIFIED` | typed plans/orchestrator와 UI state | 실제 event와 표시 상태 일치 유지 |
| 59 | [ ] | 읽기·쓰기 디렉터리 경계 | `IMPLEMENTED_UNVERIFIED` | `cogni_agent/tools.py`, path security tests | 링크/TOCTOU 경계 유지 |
| 60 | [ ] | 명령 allowlist | `IMPLEMENTED_UNVERIFIED` | `cogni_agent/tools.py`, task-plan tests | 자연어를 shell로 직접 전달하지 않기 |
| 61 | [ ] | 임의 shell·네트워크·경로 탈출 차단 | `PARTIAL` | 정책/경로 단위 테스트는 통과; `docs/SECURITY.md` | process-tree와 network를 강제하는 OS sandbox 실증 |
| 62 | [ ] | 변경 전 diff 제안 | `IMPLEMENTED_UNVERIFIED` | `cogni_flow/proposal_review.py`, 읽기 전용 `/api/evolution/proposals`와 diff dialog, `tests/test_proposal_review.py`, API/UI tests | source 적용·승인 endpoint 없이 digest/stale-base/read-only 경계 유지 |
| 63 | [ ] | 적용 전 회귀 테스트 | `PARTIAL` | 후보 평가/negative archive는 구현 | source patch 격리 실행과 전체 gate 연결 |
| 64 | [ ] | 취소·timeout·rollback | `PARTIAL` | IPC/task deadline·취소·candidate rollback tests | 실제 source/bundle 변경의 byte-identical rollback |

### G. Self-Harness 자가 거울치료 (65–75)

| ID | 체크 | 요구사항 | 상태 | 실제 근거 | 완료 승격 조건 |
|---:|:---:|---|---|---|---|
| 65 | [ ] | 오류·반복·hallucination 증거 수집 | `IMPLEMENTED_UNVERIFIED` | `cogni_flow/logdb.py`, `cogni_flow/production.py`, Phase 11 tests | 실제 실패 corpus capture ≥99% 외부 검증 |
| 66 | [ ] | 실패 원인/causal signature 분류 | `IMPLEMENTED_UNVERIFIED` | `cogni_flow/evolution.py`, `tests/test_evolution.py` | 수동 label precision 외부 검증 |
| 67 | [ ] | evolution 모드에서만 후보 생성 | `IMPLEMENTED_UNVERIFIED` | `cogni_flow/rhythm.py`, harness/orchestrator tests | inference와 동시 실행 금지 유지 |
| 68 | [ ] | 수정 후보 생성 | `IMPLEMENTED_UNVERIFIED` | `cogni_flow/local_proposer.py`, `cogni_flow/proposals.py` | 후보는 inert proposal임을 UI에 유지 |
| 69 | [ ] | 격리된 실제 patch 실행 | `NOT_IMPLEMENTED` | v0.3.2는 `proposal_only`; `README.md` | Phase 12 kernel/process/network 격리 sandbox |
| 70 | [ ] | 후보 회귀·보안 테스트 | `PARTIAL` | sealed evaluator/negative archive 단위 경로 | 실제 patch sandbox에서 전체 회귀·fault injection |
| 71 | [ ] | 실패 후보 폐기·negative archive | `IMPLEMENTED_UNVERIFIED` | `cogni_flow/proposals.py`, proposal/evolution tests | archive 무결성·보존 상한 유지 |
| 72 | [ ] | 검증 후 승격/승인 | `NOT_IMPLEMENTED` | 자동 승격은 명시적으로 차단 | 사람 승인 + attested promotion transaction |
| 73 | [ ] | 기존 코드 rollback | `NOT_IMPLEMENTED` | 후보 lifecycle rollback은 source rollback이 아님 | 실제 설치 bytes의 원자적·byte-identical rollback |
| 74 | [ ] | 증거·후보 이력 영속 저장 | `IMPLEMENTED_UNVERIFIED` | `cogni_flow/logdb.py`, proposal persistence tests | scope/content digest 검증 유지 |
| 75 | [ ] | UI가 아닌 실제 자가수정 E2E | `NOT_IMPLEMENTED` | Self-Harness UI는 proposal-only 상태 표시 | 실패 재현→patch→격리검증→승인→승격→rollback E2E |

### H. 첨부·멀티모달 (76–87)

| ID | 체크 | 요구사항 | 상태 | 실제 근거 | 완료 승격 조건 |
|---:|:---:|---|---|---|---|
| 76 | [ ] | `+` 파일/이미지 첨부 UI·API | `IMPLEMENTED_UNVERIFIED` | `cogni_demo/static/index.html`, `app.js`, `/api/workspace/attachments/add`, API/UI tests | 최신 bundle에 포함하여 E2E smoke |
| 77 | [ ] | TXT/MD/CSV/JSON 수신·UTF-8 검증 | `IMPLEMENTED_UNVERIFIED` | `cogni_demo/workspace_capabilities.py`, `tests/test_workspace_capabilities.py` | parser/크기/깊이 상한 유지 |
| 78 | [ ] | PDF 텍스트 추출·색인 | `PARTIAL` | 로컬 `pypdf` strict 추출을 Windows Job Object/POSIX rlimit의 256MiB·CPU 6초·wall 8초 격리 worker에서 실행; 128쪽/문자 상한·preview·AkasicDB 색인·재색인과 tests 구현 | PDF page 번호 provenance와 악성 PDF corpus 추가 |
| 79 | [ ] | PNG/JPG/JPEG/WEBP 수신·서명 검증 | `IMPLEMENTED_UNVERIFIED` | workspace capability MIME/signature gate와 tests | 수신 완료를 vision inference로 표시하지 않기 |
| 80 | [ ] | 첨부 목록·삭제·재색인 | `IMPLEMENTED_UNVERIFIED` | 영속 목록, delete/reindex API·UI와 `tests/test_workspace_capabilities.py`, `tests/test_workspace_capabilities_api.py`, UI contract tests | 삭제·재색인 API/UI와 인증·입력 상한 회귀 유지 |
| 81 | [ ] | 파일/총량/개수/JSON 깊이 제한 | `IMPLEMENTED_UNVERIFIED` | workspace capability constants와 boundary tests | 재시작 후 quota 우회 불가 유지 |
| 82 | [ ] | Gemma4Processor 이미지 tensor화 | `PARTIAL` | `cogni_agent/multimodal.py`와 processor tests는 구현; v0.4.0 image JSON은 historical | current manifest-bound E4B-it guard run과 pixel/byte/tensor·finite 증거 |
| 83 | [ ] | 이미지 tensor를 모델 입력에 연결 | `PARTIAL` | 고정 CPU tensor IPC와 worker/model/manager/server tests는 구현; 과거 blue-square smoke는 current scope 아님 | current exact commit image inference와 ID 86 VRAM gate |
| 84 | [ ] | 로컬 audio processor 연결 | `PARTIAL` | audio chat-template·고정 CPU tensor IPC·worker/service tests는 구현; 과거 STT smoke는 current scope 아님 | current exact commit E4B-it guard에서 16 kHz mono·finite/shape 검증 |
| 85 | [ ] | 비디오 processor 연결 | `NOT_IMPLEMENTED` | video token/config 표시는 실행 권한이 아님 | frame/sampling/time/VRAM 상한과 실제 추론 |
| 86 | [ ] | 멀티모달 VRAM 경계 검증 | `PARTIAL` | 이미지와 오디오 actual-model smoke는 PASS이나 증거 JSON에 peak allocated/reserved VRAM은 없음 | image/audio 조합을 current commit에서 16.7 GiB/finite/latency로 실측 |
| 87 | [ ] | 미지원 modality를 이유와 함께 비활성화 | `IMPLEMENTED_UNVERIFIED` | workspace capability payload, disabled UI controls, UI tests | capability 없을 때 자동 활성화 금지 |

### I. 로컬 RAG·AkasicDB (88–102)

| ID | 체크 | 요구사항 | 상태 | 실제 근거 | 완료 승격 조건 |
|---:|:---:|---|---|---|---|
| 88 | [ ] | `heosanghun/AkasicDB` bounded adapter | `IMPLEMENTED_UNVERIFIED` | `cogni_demo/workspace_capabilities.py::AkasicDBAdapter`, adapter tests | upstream 전체 서버를 실행하지 않고 adapter 경계 유지 |
| 89 | [ ] | upstream 고정 commit·핵심 파일 hash 검증 | `IMPLEMENTED_UNVERIFIED` | pinned `a6c8e8e...`, 3개 audited digest와 tests | upstream 변경 시 재감사·새 pin |
| 90 | [ ] | 첨부 자동 청킹·중복 제거·재시작 복구 | `IMPLEMENTED_UNVERIFIED` | content-addressed 중복 제거, bounded chunking, 영속 catalog/index 복구와 restart·quota·tamper tests | catalog/blob digest, 누적 quota와 재시작 index 회귀 유지 |
| 91 | [ ] | 관련성 있는 bounded 검색 | `IMPLEMENTED_UNVERIFIED` | deterministic lexical sketch와 Akasic stores, 실제 smoke/adapter tests | semantic 검색으로 과장하지 않기 |
| 92 | [ ] | 검색 근거를 모델 prompt에 주입 | `IMPLEMENTED_UNVERIFIED` | `cogni_demo/server.py`, `cogni_agent/manager.py`, manager/API tests | RAG off/no-result에서 일반 답변과 명확히 분리 |
| 93 | [ ] | 답변의 `[근거 N]` 인용 계약 | `IMPLEMENTED_UNVERIFIED` | `cogni_agent/response_quality.py`, RAG manager tests | 존재하지 않는 번호를 block/fallback |
| 94 | [ ] | source/chunk/score provenance 반환 | `IMPLEMENTED_UNVERIFIED` | RAG query payload와 UI message source rendering | 원문 위치/page까지 확대 시 schema 버전 갱신 |
| 95 | [ ] | 근거 없는 RAG 성공 표시 금지 | `IMPLEMENTED_UNVERIFIED` | `server.py` fail-closed RAG path, API tests | backend 미가동을 모델 지식으로 대체하지 않기 |
| 96 | [ ] | 잘못된 citation 차단 | `IMPLEMENTED_UNVERIFIED` | response quality citation validator와 tests | 문장별 다중 출처 회귀 추가 |
| 97 | [ ] | 문서 prompt injection 방어 | `IMPLEMENTED_UNVERIFIED` | evidence sanitization/bounded prompt, security tests | 실제 Gemma adversarial corpus 확대 |
| 98 | [ ] | 영속 provenance catalog | `IMPLEMENTED_UNVERIFIED` | atomic `attachment-catalog.v1.json`, content digest·media·index state, restart/tamper/symlink/quota tests | catalog schema·atomic replace·blob digest·host-path 비노출 유지 |
| 99 | [ ] | 삭제 시 검색 index·blob 제거 | `IMPLEMENTED_UNVERIFIED` | 성공 응답 전 검증 blob 물리 삭제, catalog/index commit, 실패 시 blob·catalog·RAG rollback; unlink/catalog fault injection과 stale retrieval 0 tests | 성공은 `deleted=true`와 `blob_deleted=true`가 모두 충족될 때만 표시하는 계약 유지 |
| 100 | [ ] | 검증된 로컬 semantic embedder | `NOT_IMPLEMENTED` | 현재 검색은 안정적 lexical projection | 모델 artifact/manifest, 품질·VRAM·라이선스 검증 |
| 101 | [ ] | RAG on/off toggle | `IMPLEMENTED_UNVERIFIED` | `static/index.html`, `app.js`, chat `rag` API flag, UI/API tests | 상태와 backend capability 일치 유지 |
| 102 | [ ] | 답변 provenance drawer | `PARTIAL` | 파일명·chunk·score 근거 표시와 첨부 preview 경로 | 문장별 클릭, PDF page/원문 위치 navigation, raw/summary 구분 |

### J. 음성 (103–110)

| ID | 체크 | 요구사항 | 상태 | 실제 근거 | 완료 승격 조건 |
|---:|:---:|---|---|---|---|
| 103 | [ ] | 마이크 입력 | `PARTIAL` | 브라우저 `getUserMedia`→16 kHz mono WAV→인증 loopback STT 경로와 UI/API contract tests | 실제 Windows 마이크 장치에서 권한·녹음·전사 브라우저 E2E |
| 104 | [ ] | Windows 마이크 권한 처리 | `PARTIAL` | 클릭 시에만 권한 요청하고 오류/취소 cleanup UI 경로 구현 | 권한 거부·철회·장치 없음·장치 전환을 실제 브라우저에서 검증 |
| 105 | [ ] | 로컬 STT | `PARTIAL` | 로컬 voice/API tests와 Gemma audio 경로는 구현; v0.4.0 TTS→STT JSON은 historical | current E4B-it GPU5 guard smoke와 다화자·잡음 품질 gate |
| 106 | [ ] | 음성→입력창 전사 | `IMPLEMENTED_UNVERIFIED` | `/api/workspace/voice/transcribe` 결과를 자동 전송 없이 편집 가능한 composer에 넣는 `app.js`, voice API/UI contract tests | transcript 편집과 명시적 전송 분리 유지 |
| 107 | [ ] | 녹음 시작·정지·취소 | `PARTIAL` | 30초/2 MiB 상한, start/stop/cancel/cleanup 상태 기계와 UI contract tests | 실제 브라우저에서 취소·권한 실패·최대시간·연속 녹음 E2E |
| 108 | [ ] | 음성 외부 전송 0 검증 | `PARTIAL` | 실제 voice evidence의 STT/TTS `external_calls=0`, 인증 loopback-only API와 fixed local runners | 완성 bundle에 대한 packet/egress 감사 |
| 109 | [ ] | 선택적 로컬 TTS | `IMPLEMENTED_UNVERIFIED` | Windows System.Speech 고정 runner, 실제 Microsoft Heami ko-KR WAV smoke, 인증 API와 play/stop/object-URL cleanup tests | 설치 voice probe·사용자 실행·stop·크기/시간 상한과 외부 호출 0 유지 |
| 110 | [ ] | 음성 미지원 이유 명시 | `IMPLEMENTED_UNVERIFIED` | capability payload와 disabled microphone tooltip/status | 지원 전까지 enabled 표시 금지 |

### K. 로컬 모델 선택 (111–117)

| ID | 체크 | 요구사항 | 상태 | 실제 근거 | 완료 승격 조건 |
|---:|:---:|---|---|---|---|
| 111 | [ ] | 로컬 모델 선택기 | `PARTIAL` | composer selector와 `/api/workspace/models/select`; 검증된 단일 모델만 표시 | 복수 manifest registry와 실제 worker 전환 |
| 112 | [ ] | 현재 선택 Gemma 상태 표시 | `IMPLEMENTED_UNVERIFIED` | workspace capability/UI model selector tests | Fact-book와 실시간 일치 유지 |
| 113 | [ ] | 로컬 모델 자동 발견 | `IMPLEMENTED_UNVERIFIED` | `cogni_demo/workspace_capabilities.py::discover_verified_local_models`, `tests/test_workspace_capabilities.py`; 명시적 absolute registry의 직접 자식만 bounded scan하고 sibling manifest·closed-world layout·고정 E4B-it fingerprint를 검증하며 symlink/reparse/UNC/URL/Hub ID를 차단 | exact-commit CPU attestation과 독립 evidence 결합; 발견 모델은 ID 116 전까지 로드 불가 유지 |
| 114 | [ ] | 모델별 modality 지원 표시 | `PARTIAL` | metadata/Fact-book에는 modality 구성과 active 구분 | 복수 모델 registry 및 실제 processor gate |
| 115 | [ ] | 모델별 VRAM 요구량 표시 | `PARTIAL` | 목표/과거 실측 표시는 있으나 모델별 현재 실측 registry 없음 | 동일 장치·prompt scope의 current measurement |
| 116 | [ ] | 안전한 unload/load 전환 | `NOT_IMPLEMENTED` | 단일 worker lifecycle만 존재 | lease drain→unload→memory check→load→rollback E2E |
| 117 | [ ] | 실제 모델과 목표 모델 구분 | `IMPLEMENTED_UNVERIFIED` | artifact/Fact-book/UI evidence rail | live verification 전 과거 모델 수치 재사용 금지 |

### L. 웹 검색·Lens.org (118–128)

| ID | 체크 | 요구사항 | 상태 | 실제 근거 | 완료 승격 조건 |
|---:|:---:|---|---|---|---|
| 118 | [ ] | 웹 검색 기본 OFF | `IMPLEMENTED_UNVERIFIED` | `WebAccessPolicy`, UI capability state, tests | 모든 세션 시작을 offline으로 유지 |
| 119 | [ ] | 사용자 명시적 online 전환 | `PARTIAL` | 권한/allowlist 정책은 구현, 실제 executor 없음 | 세션 opt-in UI·감사 로그·즉시 revoke E2E |
| 120 | [ ] | 일반 웹 검색 connector | `NOT_IMPLEMENTED` | 정책과 UI 상태만 존재 | 승인 provider, bounded schema, URL/time provenance |
| 121 | [ ] | Lens 공식 API connector | `EXTERNAL_BLOCKER` | 고정 `api.lens.org` HTTPS POST connector·4중 gate·bounded schema/retry와 mocked transport tests 구현 | 승인 Bearer token·약관/플랜으로 실제 Lens 응답 검증 |
| 122 | [ ] | Lens 특허 검색 | `EXTERNAL_BLOCKER` | `/patent/search` 고정 executor·정규화·UI/API mocked E2E 구현 | 승인 token으로 official 특허 실응답 schema/attribution 검증 |
| 123 | [ ] | Lens 학술 검색 | `EXTERNAL_BLOCKER` | `/scholarly/search` 고정 executor·정규화·UI/API mocked E2E 구현 | 승인 token으로 official 학술 실응답 schema/attribution 검증 |
| 124 | [ ] | Lens ID/DOI/특허 링크 인용 | `PARTIAL` | Lens ID·identifier normalization과 `https://lens.org/<검증 ID>` allowlist, UI link/API tests | 실제 특허·학술 응답에서 DOI/특허 ID/링크 attribution E2E와 provenance drawer 연결 |
| 125 | [ ] | Lens 결과→AkasicDB 색인 | `PARTIAL` | `LensAkasicBridge`, normalized provenance 문서와 `/search-and-index` mocked E2E tests | 승인 token 실응답의 영속 graph/index·재시작·삭제 provenance E2E |
| 126 | [ ] | 검색 query/domain/time 감사 provenance | `IMPLEMENTED_UNVERIFIED` | 고정 host/endpoint, retrieved time, query SHA-256, source-record SHA-256, Lens ID/canonical URL과 secret-redaction tests | token·원문 body 비노출, endpoint/time/query/source digest 계약 유지 |
| 127 | [ ] | token 없으면 자동 비활성화 | `IMPLEMENTED_UNVERIFIED` | `WebAccessPolicy` capability state와 tests | 토큰 값을 UI/log/prompt에 노출하지 않기 |
| 128 | [ ] | HTML scraping 금지 | `IMPLEMENTED_UNVERIFIED` | URL/endpoint 정책과 확장 로드맵 | 제품 데이터 경로를 공식 API로만 제한 |

### M. AI 워크스페이스 UI/UX (129–140)

| ID | 체크 | 요구사항 | 상태 | 실제 근거 | 완료 승격 조건 |
|---:|:---:|---|---|---|---|
| 129 | [ ] | 채팅을 핵심 화면으로 확대 | `IMPLEMENTED_UNVERIFIED` | `cogni_demo/static/app.css`, 1080p UI tests/screenshots | 최신 EXE에서 시각 smoke |
| 130 | [ ] | 입력창 sticky/항상 접근 가능 | `IMPLEMENTED_UNVERIFIED` | composer CSS/HTML과 `tests/test_demo_ui.py` | resize/zoom에서 가림 없음 확인 |
| 131 | [ ] | 불필요한 스크롤 최소화 | `IMPLEMENTED_UNVERIFIED` | workspace grid/viewport CSS | 장문은 대화 pane만 스크롤 유지 |
| 132 | [ ] | 첨부·RAG·모델·웹·음성을 composer에 배치 | `IMPLEMENTED_UNVERIFIED` | `static/index.html`, `app.js`, UI tests | 미구현 기능은 disabled 상태 유지 |
| 133 | [ ] | 근거 provenance drawer | `PARTIAL` | 메시지별 RAG source 표시만 구현 | 독립 drawer, 문장↔원문 위치 navigation |
| 134 | [ ] | READY/THINKING/STREAMING/FAILED/COMPLETE 상태 | `IMPLEMENTED_UNVERIFIED` | `app.js`, agent API/UI tests | backend terminal state와 일치 유지 |
| 135 | [ ] | 생성 취소 | `IMPLEMENTED_UNVERIFIED` | cancel API/UI, conversation transaction tests | cancel 후 worker drain/새 요청 정상 |
| 136 | [ ] | 새 대화 | `IMPLEMENTED_UNVERIFIED` | new conversation UI/API tests | pending turn 중 안전 처리 유지 |
| 137 | [ ] | 반응형 layout | `PARTIAL` | breakpoint CSS와 1080p QA | 1366×768, 1920×1080, 4K, 125–200% zoom 시각/키보드 QA |
| 138 | [ ] | 전역 가로 스크롤 금지 | `IMPLEMENTED_UNVERIFIED` | overflow CSS와 UI regression tests | 지원 해상도 matrix 유지 |
| 139 | [ ] | 접근성 | `PARTIAL` | ARIA/label/live region/disabled reason 일부 구현 | 키보드-only, focus order, screen-reader, contrast 감사 |
| 140 | [ ] | active/gated/disabled 시각 구분 | `IMPLEMENTED_UNVERIFIED` | capability chips, disabled controls, UI tests | 실행 증거 없는 `ACTIVE/VERIFIED` 금지 |

### N. 좌측 메뉴·데모 페이지 (141–148)

| ID | 체크 | 요구사항 | 상태 | 실제 근거 | 완료 승격 조건 |
|---:|:---:|---|---|---|---|
| 141 | [ ] | AI 워크스페이스 페이지 | `IMPLEMENTED_UNVERIFIED` | `static/index.html`, `app.js` | 최신 source smoke |
| 142 | [ ] | 미션 컨트롤 | `IMPLEMENTED_UNVERIFIED` | dashboard markup/data와 UI tests | 실측/목표/계획 분류 유지 |
| 143 | [ ] | 라이브 검증 | `IMPLEMENTED_UNVERIFIED` | validation page, server event stream | 현재 process 결과만 VERIFIED 승격 |
| 144 | [ ] | 시스템 설계 | `IMPLEMENTED_UNVERIFIED` | architecture page와 `docs/ARCHITECTURE.md` | 실제 capability 상태와 일치 유지 |
| 145 | [ ] | 사업 임팩트 | `IMPLEMENTED_UNVERIFIED` | business page와 `docs/COGNIBOARD_BUSINESS_DEMO_PLAN_KO.md` | 사업 목표를 실측으로 표시하지 않기 |
| 146 | [ ] | 증빙·로드맵 | `IMPLEMENTED_UNVERIFIED` | evidence/roadmap page, Fact-book | evidence class·scope 표시 유지 |
| 147 | [ ] | 사용자 매뉴얼/플레이북 | `IMPLEMENTED_UNVERIFIED` | `docs/COGNIBOARD_USER_MANUAL_PLAYBOOK_KO.md` | 신규 기능/버전에 맞춰 재생성 |
| 148 | [ ] | 3분 IR 모드 | `IMPLEMENTED_UNVERIFIED` | IR UI flow와 UI tests | 자동 진행/중단/키보드 smoke |

### O. 검증·증거 무결성 (149–159)

| ID | 체크 | 요구사항 | 상태 | 실제 근거 | 완료 승격 조건 |
|---:|:---:|---|---|---|---|
| 149 | [ ] | manifest 파일 무결성 | `IMPLEMENTED_UNVERIFIED` | `cogni_os/artifacts.py`, validation scripts/tests | load 전에 SHA 검증 유지 |
| 150 | [ ] | 모델 SHA/parameter inventory | `IMPLEMENTED_UNVERIFIED` | artifact manifest/factbook/tests | 모델 변경 시 전부 무효화·재측정 |
| 151 | [ ] | 현재 scope 실제 GPU 증거 | `PARTIAL` | v0.4.0 commit·RTX 5090 JSON은 historical; GPU5 guard와 evidence schema는 구현 | current exact commit·E4B-it·GPU5 source/model/config/device digest 결합 |
| 152 | [ ] | 현재 scope peak VRAM | `PARTIAL` | 과거 14.8469 GiB scoped canary | E4B-it 최신 source 및 target 4090에서 allocated/reserved 기록 |
| 153 | [ ] | CTS depth/residual 현재 증거 | `PARTIAL` | commit `7039152`의 E4B-it depth/residual 관측은 historical; 현재 v0.4.1 exact commit의 GPU5 guard 증거가 아님 | current exact commit·E4B-it snapshot·GPU5 device scope를 결합해 depth 100/100, residual, fallback/finite를 재실측 |
| 154 | [ ] | fallback/solver failure telemetry | `IMPLEMENTED_UNVERIFIED` | runtime validator/evidence schema | silent fallback 금지 유지 |
| 155 | [ ] | 전체 source 회귀 | `PARTIAL` | 이전 동결 후보의 `pytest` 891 passed, 6 skipped, 723 subtests와 Ruff/Node/diff 결과는 historical | 현재 clean exact commit에서 full pytest·Ruff check/format·Node syntax·`git diff --check` 원시 증거를 동일 source digest에 결합 |
| 156 | [ ] | 과거 metric을 현재값으로 표시 금지 | `IMPLEMENTED_UNVERIFIED` | `cogni_os/factbook.py`, UI tests | scope 불일치 시 값 제거 |
| 157 | [ ] | live 검증 전 `NOT VERIFIED` | `IMPLEMENTED_UNVERIFIED` | demo server/UI tests | startup 기본값 유지 |
| 158 | [ ] | measured/verified/target/plan 분리 | `IMPLEMENTED_UNVERIFIED` | `cogni_os/evidence.py`, `config/evidence.schema.json` | 모든 신규 기능 동일 taxonomy 적용 |
| 159 | [ ] | 로그·시각·버전·commit 결합 | `PARTIAL` | v0.4.0 BUILD_MANIFEST·bundle JSON은 historical; v0.4.1 evidence-bound builder는 구현 중 | current commit의 signed attestation·raw CPU/GPU evidence·artifact digest 원자 결합 |

### P. 문서·배포·사업 데모 (160–170)

| ID | 체크 | 요구사항 | 상태 | 실제 근거 | 완료 승격 조건 |
|---:|:---:|---|---|---|---|
| 160 | [ ] | CogniBoard 사용자 매뉴얼 | `IMPLEMENTED_UNVERIFIED` | `docs/COGNIBOARD_USER_MANUAL_PLAYBOOK_KO.md` | 최신 UI 캡처/버전으로 갱신 |
| 161 | [ ] | 운영 플레이북 | `IMPLEMENTED_UNVERIFIED` | 동일 문서의 실행·장애·검증 절차 | 신규 connector/voice 절차 추가 |
| 162 | [ ] | 전체 아키텍처 구조도 | `IMPLEMENTED_UNVERIFIED` | `docs/ARCHITECTURE.md`, expansion roadmap Mermaid | 실제 데이터 흐름과 동기화 |
| 163 | [ ] | Phase 1–11 계획·경계 | `IMPLEMENTED_UNVERIFIED` | `README.md`, Phase 8/9/10 문서, `docs/VALIDATION.md` | 상태 승격 시 evidence boundary 갱신 |
| 164 | [ ] | 사업계획 연계 데모 계획 | `IMPLEMENTED_UNVERIFIED` | `docs/COGNIBOARD_BUSINESS_DEMO_PLAN_KO.md` | 사업 수치 출처와 plan 배지 유지 |
| 165 | [ ] | 방산·신약·금융 데모 시나리오 | `IMPLEMENTED_UNVERIFIED` | business/demo UI·문서 | 실데이터/규제 검증 전 PoC로 표기 |
| 166 | [ ] | 현재 전체 소스 GitHub 업로드 | `PARTIAL` | v0.4.0 원격 branch·SHA 일치는 historical; 현재 v0.4.1 작업 소스의 remote exact-commit 일치는 미검증 | 검증된 clean v0.4.1 commit을 push하고 `git ls-remote`로 local/remote 40자리 SHA 일치 재확인 |
| 167 | [ ] | 최신 current-source package | `PARTIAL` | commit `5bcfbb4`의 wheel/source ZIP은 historical; 현재 evidence-bound release builder는 구현 중 | current clean exact commit에서 wheel/source ZIP을 재생성하고 source-tree·archive·raw evidence digest를 독립 검증 |
| 168 | [ ] | 최신 version/checksum/release notes | `PARTIAL` | v0.4.0 release notes·manual·SBOM·notices·manifest·checksums는 historical | current exact commit의 새 release bytes·notes·manual·SBOM·BUILD_MANIFEST·SHA256SUMS를 raw CPU/GPU evidence 및 독립 attestation에 결합 |
| 169 | [ ] | 최신 더블클릭 데모 실행파일 | `PARTIAL` | v0.4.0.0 EXE와 packaged-source smoke는 historical | current exact commit에서 evidence-bound EXE를 재생성하고 인증 loopback UI/API·정상 shutdown·cold-start smoke와 checksum을 결합 |
| 170 | [ ] | 재현 가능한 설치·실행 | `PARTIAL` | source 명령과 launcher 문서는 존재 | clean Windows 환경에서 offline 설치→manifest 검증→실행 재현 |

## 2. 미구현·부분·외부 차단 전용 체크리스트

아래 항목이 모두 `[x]`가 되기 전에는 “사용자 의도대로 모두 완성”이라고 보고하지 않는다.

### P0 — 현재 소스 무결성·RAG 영속성·배포

- [ ] **ID 8/9/167–169 (`PARTIAL`)**: v0.4.0 package·launcher/EXE·manual·release notes·
  SBOM·checksum·smoke는 historical이다. current clean exact commit에서 새 release bytes를
  생성하고 raw CPU/GPU evidence·독립 attestation·checksum·cold-start smoke를 결합한다.
- [ ] **ID 166 (`PARTIAL`)**: current v0.4.1 검증 commit을 GitHub에 push하고
  `git ls-remote`로 local/remote 40자리 SHA 일치를 확인한다.
- [ ] **ID 170**: 독립 clean Windows에서 offline 설치→manifest 검증→실행을 재현한다.
- [ ] **ID 152 (`PARTIAL`)**: 과거 RTX 5090 Laptop의 allocated VRAM 실측은 historical로
  보존됐다. current E4B-it/GPU5 scope와 목표 RTX 4090의 allocated/reserved를 각각
  재실측하고, 최종 bundle SHA 결합은 ID 159와 함께 완료한다.

### P1 — 대화·작업·근거 UX의 남은 범위

- [ ] **ID 39 (`PARTIAL`)**: 일반/기술/코드/상태/주제전환 독립 평가 corpus를 통과한다.
- [ ] **ID 54–56/61/63/64 (`PARTIAL`)**: source 수정과 test runner를 OS 격리·승인·취소·rollback
  계약으로 연결한다. 읽기 전용 proposal diff는 ID 62로 완료했다.
- [ ] **ID 78 (`PARTIAL`)**: bounded `pypdf` 추출·preview·색인과 CPU/RAM/wall 제한 별도
  process는 구현됐다. PDF page 번호 provenance·악성 문서 corpus를 추가한다.
- [ ] **ID 100 (`NOT_IMPLEMENTED`)**: 검증된 로컬 semantic embedder를 manifest/VRAM/품질/라이선스와 함께 도입한다.
- [ ] **ID 102/133 (`PARTIAL`)**: 문장별 citation에서 원문 파일/page/chunk로 이동하는 근거 drawer를 완성한다.
- [ ] **ID 137/139 (`PARTIAL`)**: 해상도·zoom·키보드·screen-reader·contrast 접근성 감사를 통과한다.

### P2 — 실제 멀티모달·음성·모델 전환

- [ ] **ID 85 (`NOT_IMPLEMENTED`), 86 (`PARTIAL`)**: image/audio processor·IPC와 각 1건 actual
  smoke는 구현됐다. video 입력과 modality별 peak VRAM/latency/품질 matrix를 검증한다.
- [ ] **ID 103/104/107/108 (`PARTIAL`)**: 로컬 STT·composer 전사·TTS는 구현됐다. 실제 Windows
  마이크에서 권한 거부/철회·녹음 stop/cancel·연속 실행과 완성 bundle packet audit를 검증한다.
- [ ] **ID 113 (`IMPLEMENTED_UNVERIFIED`)**: bounded 복수 manifest 발견 경로는 구현됐다. exact-commit
  CPU attestation과 독립 evidence를 결합한다.
- [ ] **ID 111/114/115 (`PARTIAL`), 116 (`NOT_IMPLEMENTED`)**: 발견된 model의 modality/VRAM
  상태와 lease-safe unload/load/rollback을 구현한다.

### P3 — 온라인 연구·Lens

- [ ] **ID 119 (`PARTIAL`)**: 세션별 online opt-in/revoke를 구현한다. Lens query/domain/time/source
  digest와 secret redaction은 ID 126으로 완료했다.
- [ ] **ID 120 (`NOT_IMPLEMENTED`), 124/125 (`PARTIAL`)**: 일반 검색 provider는 없고 Lens의
  검증 링크·AkasicDB bridge는 mocked E2E까지다. 실제 응답 provenance와 영속 index를 검증한다.
- [ ] **ID 121–123 (`EXTERNAL_BLOCKER`)**: 고정 Lens 공식 API connector는 구현됐다. 승인된 Bearer
  token과 사용 약관/상용 플랜으로 특허·학술 실응답을 검증한다.

### P4 — 연구 모듈 승격과 실제 Self-Harness

- [ ] **ID 10/14/19/20/22–29/31/32 (`PARTIAL`)**: 학습 artifact, 독립 held-out 평가,
  calibration 및 현재 scope 실측 없이 `active`로 승격하지 않는다.
- [ ] **ID 69/72/73/75 (`NOT_IMPLEMENTED`), 70 (`PARTIAL`)**: Phase 12 kernel-isolated sandbox,
  실제 patch 회귀, 사람 승인, 원자적 승격, byte-identical rollback E2E를 구현한다.
- [ ] **ID 4/16 (`EXTERNAL_BLOCKER`)**: 정확한 RTX 4090 24GB에서 최신 E4B-it 통합 gate를 실행한다.
- [ ] **ID 6 (`PARTIAL`)**: 완성 배포물의 packet/egress 및 공급망/오프라인 의존성을 외부 감사한다.

## 3. 우선순위와 의존성

```text
P0-1 영속 첨부/RAG·삭제
  └─> P0-2 전체 source 회귀
       └─> P0-3 실제 Gemma/현재 GPU 검증
            └─> P0-4 버전 동결·EXE/package/checksum·GitHub

P1 PDF/semantic RAG/근거 drawer
  └─> P2 Gemma4Processor tensor IPC
       ├─> vision/audio/video VRAM gate
       ├─> local STT/TTS
       └─> 복수 모델 safe switch

P3 online session policy
  └─> 승인 provider/Lens token·terms
       └─> Lens record normalization
            └─> AkasicDB provenance index·citation drawer

P4 attested sandbox
  └─> 실제 patch 평가
       └─> 사람 승인·원자 승격
            └─> byte-identical rollback
```

현재 장치에서 코드로 준비할 수 있는 P0 배포 후보 경로는 구현됐지만, current exact-commit
CPU/GPU 증거 결합·새 패키지·EXE smoke·GitHub SHA 일치가 남아 있어 완료가 아니다. 독립 clean
Windows 재현과 목표 RTX 4090 검증은 현재 장치 밖의 별도 gate다. 멀티모달·음성은 입력 버튼보다 모델 tensor 계약과
VRAM gate가 선행되어야 한다. Lens는 connector 코드뿐 아니라 승인 토큰·약관·감사 경계가 함께
준비되어야 한다. Self-Harness 자동 승격은 OS 수준 격리와 rollback 증거보다 먼저 열 수 없다.

## 4. 과장 금지 및 완료 보고 규칙

1. **구현, 연결, 검증, 배포를 구분한다.** 클래스나 버튼이 존재해도 worker와 제품 API로
   연결되지 않았으면 완료가 아니다.
2. **현재 scope만 현재 증거다.** 모델·manifest·commit·config·device 중 하나가 바뀌면 과거
   GPU/대화/회귀 수치를 현재값으로 재사용하지 않는다.
3. **target과 measured를 바꾸지 않는다.** RTX 4090과 16.7 GiB는 목표이며, RTX 5090 Laptop
   실측은 목표 장치 인증이 아니다.
4. **checkpoint metadata를 capability로 부르지 않는다.** 이미지·오디오는 실제 tensor IPC와
   제한된 smoke 증거 범위만 표시하며, 비디오는 processor 설정만 있으므로 비활성이다.
5. **lexical RAG를 semantic RAG로 부르지 않는다.** 현재 AkasicDB adapter의 안정적 lexical
   projection은 유용하지만 검증된 embedding model은 아니다.
6. **외부 검색을 시뮬레이션하지 않는다.** Lens token·공식 API 응답·provenance가 없으면
   `인증 필요/미구현`으로 표시하며 HTML scraping이나 모델 기억을 검색 결과로 대체하지 않는다.
7. **proposal을 자가수정으로 부르지 않는다.** 현재 Self-Harness는 inert proposal과 negative
   archive까지이며 실제 source 설치·자동 승격·source rollback은 없다.
8. **단위 테스트를 사업/품질 인증으로 확대하지 않는다.** 답변 품질, trained module 효과,
   공인 시험, 라이선스, 보안은 각자의 외부 증거가 필요하다.
9. **최신 실행파일의 byte provenance를 확인한다.** 같은 동결 commit에서 생성한 EXE·source
   archive·manual·checksum만 해당 릴리스 증거로 인정한다.
10. **완료 문구는 이 원장의 모든 비완료 항목이 승격된 뒤 사용한다.** 그 전에는 현재 완료,
    부분, 미구현, 외부 차단 수와 다음 gate를 함께 보고한다.
