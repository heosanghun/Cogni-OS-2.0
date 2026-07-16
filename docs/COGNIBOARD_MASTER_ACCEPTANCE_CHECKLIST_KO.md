# CogniBoard 마스터 완료 기준 체크리스트 (ID 1–170)

## 0. 문서 목적과 판정 기준

이 문서는 사용자가 요청한 기능을 빠짐없이 고정 ID로 추적하는 단일 완료 원장이다. 판정 기준은
`agent/workspace-rag-multimodal-v0.4.0` 브랜치의 기준 커밋 `e090b3a` 이후 현재 v0.4.0
개발 소스이다. 항목별 상태는 현재 코드 경로와 해당 범위의 자동 검증 또는 실측 증거만 반영한다.
현재 패키지·커밋·푸시는 아직 끝나지 않았으므로 그 범위의 항목은 승격하지 않는다.
과거 v0.3.2 EXE와 과거 검증 JSON은 최신 개발 소스의 완료 증거가 아니다.

상태의 의미는 다음과 같다.

| 상태 | 의미 |
|---|---|
| `COMPLETED` | 현재 소스에 실행 경로가 연결되어 있고, 해당 범위의 자동 테스트 또는 동일 범위 실측 증거가 있다. |
| `PARTIAL` | 일부 경로는 있으나 제품 연결, 독립 검증, 현재 소스 회귀 또는 요구 범위 일부가 남았다. |
| `NOT_IMPLEMENTED` | 제품에서 실행 가능한 경로가 없거나 의도적으로 차단되어 있다. |
| `EXTERNAL_BLOCKER` | 코드만으로 승격할 수 없으며 지정 하드웨어·승인 토큰·라이선스·외부 독립 평가가 필요하다. |

`[x]`는 위 정의의 `COMPLETED`만 뜻한다. `PARTIAL`이나 외부 자료가 없는 상태를 화면 표시,
설계 문서, 단위 테스트만으로 완료라고 부르지 않는다.

현재 스냅샷 집계는 `COMPLETED 108 / PARTIAL 44 / NOT_IMPLEMENTED 13 /
EXTERNAL_BLOCKER 5`이다. 즉 170개 중 62개는 아직 승격 조건이 남아 있다. 실제 20-turn
대화 stress, 전체 source 회귀, 동결 커밋 기준 CTS 재실측은 통과했으며 최신 패키지·EXE·
GitHub 업로드는 아직 이 집계에서 완료로 보지 않는다.

## 1. 전체 기능 체크리스트

### A. 기본 런타임과 제품 경계 (1–9)

| ID | 체크 | 요구사항 | 상태 | 실제 근거 | 완료 승격 조건 |
|---:|:---:|---|---|---|---|
| 1 | [x] | Gemma 4 E4B-it 로컬 백본 | `COMPLETED` | `cogni_agent/model_service.py`, `config/gemma4-e4b-it.manifest.toml`, `tests/test_agent_model_service.py` | 현재 동결 아티팩트·manifest 계약 유지 |
| 2 | [x] | effective 파라미터 약 4.5B 표시 | `COMPLETED` | `cogni_os/artifacts.py`, `cogni_os/factbook.py`, `tests/test_artifacts.py` | 모델 교체 시 실제 텐서 inventory 재계산 |
| 3 | [x] | 저장 파라미터와 effective 파라미터 구분 | `COMPLETED` | `cogni_os/artifacts.py`, `tests/test_agent_fact_grounding.py` | Fact-book 수치와 manifest 일치 유지 |
| 4 | [ ] | RTX 4090 24GB 목표 장치 검증 | `EXTERNAL_BLOCKER` | 목표는 `config/default.toml`; 기존 실측은 `release/evidence/phase11_gemma_cts_runtime_20260712.json`의 RTX 5090 Laptop | 정확한 RTX 4090에서 동결 소스·E4B-it 전체 게이트 재실측 |
| 5 | [x] | 실제 장치와 목표 장치 구분 표시 | `COMPLETED` | `cogni_os/factbook.py`, `cogni_demo/static/app.js`, `tests/test_factbook.py` | 라이브 검증 전에는 `NOT VERIFIED` 유지 |
| 6 | [ ] | 기본 air-gap 및 외부 호출 0 | `PARTIAL` | `cogni_os/runtime.py`, `cogni_demo/server.py`, `tests/test_airgap.py` | 현재 배포 바이트에 대한 패킷/egress 감사와 오프라인 의존성 검토 |
| 7 | [x] | 외부 API는 명시적 온라인 모드에서만 허용 | `COMPLETED` | `cogni_demo/workspace_capabilities.py::WebAccessPolicy`, `tests/test_workspace_capabilities.py` | 모든 향후 connector가 동일 권한 계약 사용 |
| 8 | [ ] | 최신 기능 통합 더블클릭 EXE | `NOT_IMPLEMENTED` | 기존 `CogniBoard.exe`/v0.3.2 bundle은 `e090b3a` 이후 기능을 포함하지 않음 | 동결한 최신 커밋에서 launcher·bundle 재생성 및 smoke |
| 9 | [ ] | 패키지된 EXE 기능 회귀 | `NOT_IMPLEMENTED` | `tests/test_release_bundle.py`는 기존 bundle 계약만 검증 | 새 EXE로 첨부·RAG·UI·대화·검증 E2E 실행 |

### B. Cogni-Core: DEQ·CTS (10–18)

| ID | 체크 | 요구사항 | 상태 | 실제 근거 | 완료 승격 조건 |
|---:|:---:|---|---|---|---|
| 10 | [ ] | Gemma 특징과 DEQ 고정점 경로 결합 | `PARTIAL` | `cogni_agent/core_pipeline.py`, `cogni_core/deq.py`, `tests/test_core_pipeline.py` | 학습된 DEQ/Wproj와 held-out 인과 ablation으로 canary 해제 |
| 11 | [x] | Broyden/L-Broyden bounded solver | `COMPLETED` | `cogni_core/deq.py`, `cogni_core/search.py`, `tests/test_deq.py`, `tests/test_search.py` | rank/history 상한 회귀 유지 |
| 12 | [x] | CTS Depth 100 이상 | `COMPLETED` | `cogni_core/cts.py`, `scripts/validate_gemma4_runtime.py`, Phase 11 runtime JSON | 같은 범위에서 100/100·finite 유지 |
| 13 | [x] | 고정 301-node arena | `COMPLETED` | `cogni_core/cts.py`, `tests/test_cts_policy.py` | arena 초과 fail-closed 유지 |
| 14 | [ ] | 추론 깊이에 선형 증가하지 않는 메모리 | `PARTIAL` | CTS arena와 solver history는 고정; `docs/VALIDATION.md` | 모델·로그·외부 데이터 전체 O(1) 주장은 금지하고 심도별 실제 곡선 재측정 |
| 15 | [x] | KV-cache 비축 금지 | `COMPLETED` | `cogni_agent/model_service.py`의 `use_cache=False`, `tests/test_agent_model_service.py` | 모든 decode 경로에서 cache 비활성 유지 |
| 16 | [ ] | 16.7 GiB VRAM 방어선 | `EXTERNAL_BLOCKER` | `scripts/validate_gemma4_runtime.py` postcondition; 과거 5090 실측 14.8469 GiB | 목표 RTX 4090·E4B-it·현재 소스에서 allocated/reserved 모두 재검증 |
| 17 | [x] | residual·finite·fallback 명시 | `COMPLETED` | `cogni_core/deq.py`, `cogni_os/evidence.py`, runtime validator/tests | 실패 시 답변 미게시 계약 유지 |
| 18 | [x] | 실제 GPU 메모리/장치 프로파일링 | `COMPLETED` | `scripts/validate_gemma4_runtime.py`, Phase 11 runtime JSON | 새 릴리스마다 현재 scope 증거 갱신 |

### C. System 1.5·2.5·3·4 (19–30)

| ID | 체크 | 요구사항 | 상태 | 실제 근거 | 완료 승격 조건 |
|---:|:---:|---|---|---|---|
| 19 | [ ] | System 1.5 Fast Weight | `PARTIAL` | `cogni_core/fast_weights.py`, `tests/test_fastweight_runtime.py`; Fact-book `gated` | 승인된 학습 checkpoint와 AQ/OOD·비회귀 증거 |
| 20 | [ ] | DEQ 상태를 저랭크 임시 가중치로 변환 | `PARTIAL` | `cogni_core/fast_weights.py`, `cogni_core/adaptation.py` | 실제 학습 artifact로 answer-quality 이득 검증 |
| 21 | [x] | 검증 세션에서만 Fast Weight 활성 | `COMPLETED` | `cogni_core/fast_weight_safety.py`, `tests/test_fast_weight_safety.py` | TTL/LRU/session/artifact binding 유지 |
| 22 | [ ] | System 2.5 FP-EWC | `PARTIAL` | `cogni_core/fp_ewc.py`, `tests/test_fp_ewc.py`; `night_only` | 실제 도메인 3-seed BWT/FWT 증거 |
| 23 | [ ] | Fisher 기반 망각 방지 | `PARTIAL` | empirical Fisher/anchor/merge 구현과 `tests/test_fp_ewc.py` | 독립 데이터셋의 망각 감소와 품질 비회귀 |
| 24 | [ ] | C-FIRE spectral norm < 0.95 | `PARTIAL` | `cogni_core/c_fire.py`, `tests/test_c_fire.py` | 전체 적용 대상과 합성 연산의 인증서·실측; 전체 decoder 수축 주장 금지 |
| 25 | [ ] | System 3 Sparse MoE | `PARTIAL` | `cogni_core/experts.py`, `cogni_core/expert_lifecycle.py`, Phase 8 tests | 학습·교정 expert checkpoint와 독립 verifier |
| 26 | [ ] | novelty 기반 routing | `PARTIAL` | `cogni_core/routing.py`, `tests/test_routing.py` | 실제 OOD corpus threshold calibration |
| 27 | [ ] | 안전한 expert spawn·등록·rollback | `PARTIAL` | `cogni_core/expert_lifecycle.py`, `tests/test_expert_lifecycle.py` | 제품 artifact promotion의 독립 검증 |
| 28 | [ ] | System 4 Tensor Swarm | `PARTIAL` | `cogni_core/swarm.py`, `cogni_core/swarm_sessions.py`, `tests/test_swarm.py`; `advisory` | production PCAS corpus와 답변 품질 ablation |
| 29 | [ ] | 내부 에이전트 tensor 중심 통신 | `PARTIAL` | System 4 tensor 경로와 `cogni_os/tensor_service.py`; 제어/제품 전체가 tensor-only는 아님 | 모든 대상 경계의 명시적 tensor schema·복사/직렬화 측정 |
| 30 | [x] | System 4 latency 검증 | `COMPLETED` | `scripts/benchmark_system4.py`, `release/evidence/phase11_system4_stress_20260712.json` | 환경 변경 시 p50/p95/p99와 수렴/finite 재측정 |

### D. Cogni-Flow·오케스트레이션 (31–37)

| ID | 체크 | 요구사항 | 상태 | 실제 근거 | 완료 승격 조건 |
|---:|:---:|---|---|---|---|
| 31 | [ ] | BIO-HAMA meta-router | `PARTIAL` | `cogni_core/meta_router.py`, `tests/test_meta_router.py`; `advisory` | 실제 routing quality calibration |
| 32 | [ ] | ADAS/AFlow pipeline 생성·탐색 | `PARTIAL` | `cogni_flow/aflow.py`, `cogni_flow/aflow_research.py`; `research_archive_only` | attested evaluator와 제품 설치 권한 분리 검증 |
| 33 | [x] | 계획→실행→검토→재계획 루프 | `COMPLETED` | `cogni_flow/orchestrator.py`, `cogni_flow/task_plan.py`, 관련 tests | bounded 횟수·deadline 유지 |
| 34 | [x] | bounded MCTS/workflow search | `COMPLETED` | `cogni_flow/aflow.py`, `tests/test_aflow.py` | archive/후보/노드 상한 유지 |
| 35 | [x] | CPU control plane / GPU data plane 분리 | `COMPLETED` | `docs/ARCHITECTURE.md`, `cogni_os/runtime.py` | 단일 GPU owner 계약 유지 |
| 36 | [x] | 단일 GPU lease owner | `COMPLETED` | `cogni_os/gpu_lease.py`, `tests/test_gpu_lease.py` | epoch/job/deadline binding 유지 |
| 37 | [x] | 주간 inference/야간 evolution 상호 배제 | `COMPLETED` | `cogni_flow/rhythm.py`, `tests/test_rhythm.py` | 동시 진입 fail-closed 유지 |

### E. 자연스러운 대화 품질 (38–52)

| ID | 체크 | 요구사항 | 상태 | 실제 근거 | 완료 승격 조건 |
|---:|:---:|---|---|---|---|
| 38 | [x] | 하드코딩이 아닌 로컬 Gemma 생성 | `COMPLETED` | `cogni_agent/model_service.py`, `cogni_agent/manager.py`, actual-model validation JSON | 모델 실패를 템플릿 정상 답변으로 위장하지 않기 |
| 39 | [ ] | 임의의 일반 질문 응답 | `PARTIAL` | 실제 Gemma 대화 경로 존재; typed tool 권한은 제한 | 독립 human-labelled 다양한 주제 acceptance |
| 40 | [x] | 자연스러운 한국어 대화 | `COMPLETED` | `scripts/validate_agent_casual_korean.py`, `validation/evidence/agent_casual_korean_v040.json`의 현재 v0.4.0 10/10 PASS | 현재 소스/모델 10-turn gate 유지 |
| 41 | [x] | bounded 다중 turn 문맥 | `COMPLETED` | `cogni_agent/conversation.py`, `tests/test_agent_conversation.py` | 압축/제거 시 turn 무결성 유지 |
| 42 | [x] | 주제 전환 처리 | `COMPLETED` | conversation/quality integration tests | 주제어 억지 복사 없이 회귀 유지 |
| 43 | [x] | 동일 문장 무한 반복 방지 | `COMPLETED` | `cogni_agent/response_quality.py`, completion stress evidence | 최신 소스 실모델 stress 재실행 |
| 44 | [x] | 답변 중간 끊김 방지 | `COMPLETED` | `cogni_agent/manager.py`, `tests/test_agent_completion_stress.py` | terminal stop/timeout/final sentence gate 유지 |
| 45 | [x] | 완결 문장 게시 | `COMPLETED` | response quality tests | 형식 요청과 자연문장을 구분해 strict gate 유지 |
| 46 | [x] | 길이 한계 후 이어쓰기 | `COMPLETED` | `cogni_agent/conversation.py`, manager continuation tests | 중복 prefix 없는 동일 turn 연속성 유지 |
| 47 | [x] | USER/ASSISTANT/control token 누출 방지 | `COMPLETED` | `cogni_agent/manager.py`, `tests/test_response_quality.py` | 신규 tokenizer special token 회귀 추가 |
| 48 | [x] | 사용자 질문 echo 방지 | `COMPLETED` | manager echo stripping과 `tests/test_agent_manager.py` | 인용 요청은 오탐 제거하지 않기 |
| 49 | [x] | 모델 정체성·파라미터 hallucination 방지 | `COMPLETED` | `cogni_agent/fact_grounding.py`, `tests/test_agent_fact_grounding.py` | Fact-book 외 수치 생성 차단 유지 |
| 50 | [x] | 아키텍처·상태를 실제 capability로 설명 | `COMPLETED` | `cogni_os/capabilities.py`, Fact-book, grounding tests | `gated/advisory`를 active로 표현하지 않기 |
| 51 | [x] | 질문당 assistant 답변 정확히 1개 | `COMPLETED` | transactional manager/conversation tests, validation addendum | 재시도 후보를 UI에 중복 게시하지 않기 |
| 52 | [x] | 현재 소스 actual-model 품질 stress | `COMPLETED` | `validation/evidence/agent_casual_korean_v040.json` 10/10 및 `validation/evidence/agent_completion_20turn_v040.json` 20/20 PASS; 품질 fallback 0, 문장 반복률 0 | 모델·prompt 계약 변경 시 동일 corpus와 strict gate 재실행 |

### F. 로컬 작업 에이전트 (53–64)

| ID | 체크 | 요구사항 | 상태 | 실제 근거 | 완료 승격 조건 |
|---:|:---:|---|---|---|---|
| 53 | [x] | 프로젝트 파일 읽기·검색 | `COMPLETED` | `cogni_agent/tools.py`, `tests/test_agent_tools.py` | root/path/reparse/size gate 유지 |
| 54 | [ ] | 코드 작성·수정 | `PARTIAL` | output-only write와 T2 proposal staging; 실제 source mutation 금지 | 승인·격리·회귀·rollback이 결합된 source patch 경로 |
| 55 | [ ] | PoC/MVP 개발 지원 | `PARTIAL` | `cogni_agent/tools.py`의 bounded `/project` 다중 파일 묶음, `cogni_flow/task_plan.py`, `tests/test_agent_project_bundle.py`; output-only·무실행·무덮어쓰기 | 자연어 요청→검토 가능한 typed bundle 생성과 선택적 격리 테스트 E2E |
| 56 | [ ] | 테스트 실행 | `PARTIAL` | 고정 pytest primitive, `cogni_flow/task_plan.py`; 기본 차단 | OS 격리와 명시적 trusted opt-in |
| 57 | [x] | 결과·실패 로그 저장 | `COMPLETED` | `cogni_flow/logdb.py`, `tests/test_logdb.py` | 민감정보/크기/보존정책 유지 |
| 58 | [x] | 계획·진행률 표시 | `COMPLETED` | typed plans/orchestrator와 UI state | 실제 event와 표시 상태 일치 유지 |
| 59 | [x] | 읽기·쓰기 디렉터리 경계 | `COMPLETED` | `cogni_agent/tools.py`, path security tests | 링크/TOCTOU 경계 유지 |
| 60 | [x] | 명령 allowlist | `COMPLETED` | `cogni_agent/tools.py`, task-plan tests | 자연어를 shell로 직접 전달하지 않기 |
| 61 | [ ] | 임의 shell·네트워크·경로 탈출 차단 | `PARTIAL` | 정책/경로 단위 테스트는 통과; `docs/SECURITY.md` | process-tree와 network를 강제하는 OS sandbox 실증 |
| 62 | [x] | 변경 전 diff 제안 | `COMPLETED` | `cogni_flow/proposal_review.py`, 읽기 전용 `/api/evolution/proposals`와 diff dialog, `tests/test_proposal_review.py`, API/UI tests | source 적용·승인 endpoint 없이 digest/stale-base/read-only 경계 유지 |
| 63 | [ ] | 적용 전 회귀 테스트 | `PARTIAL` | 후보 평가/negative archive는 구현 | source patch 격리 실행과 전체 gate 연결 |
| 64 | [ ] | 취소·timeout·rollback | `PARTIAL` | IPC/task deadline·취소·candidate rollback tests | 실제 source/bundle 변경의 byte-identical rollback |

### G. Self-Harness 자가 거울치료 (65–75)

| ID | 체크 | 요구사항 | 상태 | 실제 근거 | 완료 승격 조건 |
|---:|:---:|---|---|---|---|
| 65 | [x] | 오류·반복·hallucination 증거 수집 | `COMPLETED` | `cogni_flow/logdb.py`, `cogni_flow/production.py`, Phase 11 tests | 실제 실패 corpus capture ≥99% 외부 검증 |
| 66 | [x] | 실패 원인/causal signature 분류 | `COMPLETED` | `cogni_flow/evolution.py`, `tests/test_evolution.py` | 수동 label precision 외부 검증 |
| 67 | [x] | evolution 모드에서만 후보 생성 | `COMPLETED` | `cogni_flow/rhythm.py`, harness/orchestrator tests | inference와 동시 실행 금지 유지 |
| 68 | [x] | 수정 후보 생성 | `COMPLETED` | `cogni_flow/local_proposer.py`, `cogni_flow/proposals.py` | 후보는 inert proposal임을 UI에 유지 |
| 69 | [ ] | 격리된 실제 patch 실행 | `NOT_IMPLEMENTED` | v0.3.2는 `proposal_only`; `README.md` | Phase 12 kernel/process/network 격리 sandbox |
| 70 | [ ] | 후보 회귀·보안 테스트 | `PARTIAL` | sealed evaluator/negative archive 단위 경로 | 실제 patch sandbox에서 전체 회귀·fault injection |
| 71 | [x] | 실패 후보 폐기·negative archive | `COMPLETED` | `cogni_flow/proposals.py`, proposal/evolution tests | archive 무결성·보존 상한 유지 |
| 72 | [ ] | 검증 후 승격/승인 | `NOT_IMPLEMENTED` | 자동 승격은 명시적으로 차단 | 사람 승인 + attested promotion transaction |
| 73 | [ ] | 기존 코드 rollback | `NOT_IMPLEMENTED` | 후보 lifecycle rollback은 source rollback이 아님 | 실제 설치 bytes의 원자적·byte-identical rollback |
| 74 | [x] | 증거·후보 이력 영속 저장 | `COMPLETED` | `cogni_flow/logdb.py`, proposal persistence tests | scope/content digest 검증 유지 |
| 75 | [ ] | UI가 아닌 실제 자가수정 E2E | `NOT_IMPLEMENTED` | Self-Harness UI는 proposal-only 상태 표시 | 실패 재현→patch→격리검증→승인→승격→rollback E2E |

### H. 첨부·멀티모달 (76–87)

| ID | 체크 | 요구사항 | 상태 | 실제 근거 | 완료 승격 조건 |
|---:|:---:|---|---|---|---|
| 76 | [x] | `+` 파일/이미지 첨부 UI·API | `COMPLETED` | `cogni_demo/static/index.html`, `app.js`, `/api/workspace/attachments/add`, API/UI tests | 최신 bundle에 포함하여 E2E smoke |
| 77 | [x] | TXT/MD/CSV/JSON 수신·UTF-8 검증 | `COMPLETED` | `cogni_demo/workspace_capabilities.py`, `tests/test_workspace_capabilities.py` | parser/크기/깊이 상한 유지 |
| 78 | [ ] | PDF 텍스트 추출·색인 | `PARTIAL` | 로컬 `pypdf` strict 추출을 Windows Job Object/POSIX rlimit의 256MiB·CPU 6초·wall 8초 격리 worker에서 실행; 128쪽/문자 상한·preview·AkasicDB 색인·재색인과 tests 구현 | PDF page 번호 provenance와 악성 PDF corpus 추가 |
| 79 | [x] | PNG/JPG/JPEG/WEBP 수신·서명 검증 | `COMPLETED` | workspace capability MIME/signature gate와 tests | 수신 완료를 vision inference로 표시하지 않기 |
| 80 | [x] | 첨부 목록·삭제·재색인 | `COMPLETED` | 영속 목록, delete/reindex API·UI와 `tests/test_workspace_capabilities.py`, `tests/test_workspace_capabilities_api.py`, UI contract tests | 삭제·재색인 API/UI와 인증·입력 상한 회귀 유지 |
| 81 | [x] | 파일/총량/개수/JSON 깊이 제한 | `COMPLETED` | workspace capability constants와 boundary tests | 재시작 후 quota 우회 불가 유지 |
| 82 | [x] | Gemma4Processor 이미지 tensor화 | `COMPLETED` | `cogni_agent/multimodal.py`, `tests/test_multimodal_processor.py`, 실제 smoke `validation/evidence/gemma4_local_image_v040.json` | manifest-bound processor, pixel/byte/tensor 상한과 finite 검증 유지 |
| 83 | [x] | 이미지 tensor를 모델 입력에 연결 | `COMPLETED` | 고정 CPU tensor IPC, worker/model/manager/server 연결과 IPC/API tests; 실제 파란 정사각형 smoke PASS | smoke 결과를 일반 vision 품질로 확대하지 않고 별도 ID 86 VRAM gate 유지 |
| 84 | [x] | 로컬 audio processor 연결 | `COMPLETED` | `Gemma4Processor` audio chat-template, 고정 CPU tensor IPC, model worker/voice service 연결과 `tests/test_audio_worker_ipc.py`; 실제 STT smoke PASS | 16 kHz mono PCM·manifest·finite/shape 계약 유지 |
| 85 | [ ] | 비디오 processor 연결 | `NOT_IMPLEMENTED` | video token/config 표시는 실행 권한이 아님 | frame/sampling/time/VRAM 상한과 실제 추론 |
| 86 | [ ] | 멀티모달 VRAM 경계 검증 | `PARTIAL` | 이미지와 오디오 actual-model smoke는 PASS이나 증거 JSON에 peak allocated/reserved VRAM은 없음 | image/audio 조합을 current commit에서 16.7 GiB/finite/latency로 실측 |
| 87 | [x] | 미지원 modality를 이유와 함께 비활성화 | `COMPLETED` | workspace capability payload, disabled UI controls, UI tests | capability 없을 때 자동 활성화 금지 |

### I. 로컬 RAG·AkasicDB (88–102)

| ID | 체크 | 요구사항 | 상태 | 실제 근거 | 완료 승격 조건 |
|---:|:---:|---|---|---|---|
| 88 | [x] | `heosanghun/AkasicDB` bounded adapter | `COMPLETED` | `cogni_demo/workspace_capabilities.py::AkasicDBAdapter`, adapter tests | upstream 전체 서버를 실행하지 않고 adapter 경계 유지 |
| 89 | [x] | upstream 고정 commit·핵심 파일 hash 검증 | `COMPLETED` | pinned `a6c8e8e...`, 3개 audited digest와 tests | upstream 변경 시 재감사·새 pin |
| 90 | [x] | 첨부 자동 청킹·중복 제거·재시작 복구 | `COMPLETED` | content-addressed 중복 제거, bounded chunking, 영속 catalog/index 복구와 restart·quota·tamper tests | catalog/blob digest, 누적 quota와 재시작 index 회귀 유지 |
| 91 | [x] | 관련성 있는 bounded 검색 | `COMPLETED` | deterministic lexical sketch와 Akasic stores, 실제 smoke/adapter tests | semantic 검색으로 과장하지 않기 |
| 92 | [x] | 검색 근거를 모델 prompt에 주입 | `COMPLETED` | `cogni_demo/server.py`, `cogni_agent/manager.py`, manager/API tests | RAG off/no-result에서 일반 답변과 명확히 분리 |
| 93 | [x] | 답변의 `[근거 N]` 인용 계약 | `COMPLETED` | `cogni_agent/response_quality.py`, RAG manager tests | 존재하지 않는 번호를 block/fallback |
| 94 | [x] | source/chunk/score provenance 반환 | `COMPLETED` | RAG query payload와 UI message source rendering | 원문 위치/page까지 확대 시 schema 버전 갱신 |
| 95 | [x] | 근거 없는 RAG 성공 표시 금지 | `COMPLETED` | `server.py` fail-closed RAG path, API tests | backend 미가동을 모델 지식으로 대체하지 않기 |
| 96 | [x] | 잘못된 citation 차단 | `COMPLETED` | response quality citation validator와 tests | 문장별 다중 출처 회귀 추가 |
| 97 | [x] | 문서 prompt injection 방어 | `COMPLETED` | evidence sanitization/bounded prompt, security tests | 실제 Gemma adversarial corpus 확대 |
| 98 | [x] | 영속 provenance catalog | `COMPLETED` | atomic `attachment-catalog.v1.json`, content digest·media·index state, restart/tamper/symlink/quota tests | catalog schema·atomic replace·blob digest·host-path 비노출 유지 |
| 99 | [x] | 삭제 시 검색 index·blob 제거 | `COMPLETED` | 성공 응답 전 검증 blob 물리 삭제, catalog/index commit, 실패 시 blob·catalog·RAG rollback; unlink/catalog fault injection과 stale retrieval 0 tests | 성공은 `deleted=true`와 `blob_deleted=true`가 모두 충족될 때만 표시하는 계약 유지 |
| 100 | [ ] | 검증된 로컬 semantic embedder | `NOT_IMPLEMENTED` | 현재 검색은 안정적 lexical projection | 모델 artifact/manifest, 품질·VRAM·라이선스 검증 |
| 101 | [x] | RAG on/off toggle | `COMPLETED` | `static/index.html`, `app.js`, chat `rag` API flag, UI/API tests | 상태와 backend capability 일치 유지 |
| 102 | [ ] | 답변 provenance drawer | `PARTIAL` | 파일명·chunk·score 근거 표시와 첨부 preview 경로 | 문장별 클릭, PDF page/원문 위치 navigation, raw/summary 구분 |

### J. 음성 (103–110)

| ID | 체크 | 요구사항 | 상태 | 실제 근거 | 완료 승격 조건 |
|---:|:---:|---|---|---|---|
| 103 | [ ] | 마이크 입력 | `PARTIAL` | 브라우저 `getUserMedia`→16 kHz mono WAV→인증 loopback STT 경로와 UI/API contract tests | 실제 Windows 마이크 장치에서 권한·녹음·전사 브라우저 E2E |
| 104 | [ ] | Windows 마이크 권한 처리 | `PARTIAL` | 클릭 시에만 권한 요청하고 오류/취소 cleanup UI 경로 구현 | 권한 거부·철회·장치 없음·장치 전환을 실제 브라우저에서 검증 |
| 105 | [x] | 로컬 STT | `COMPLETED` | 동일 manifest-bound Gemma audio 경로와 `validation/evidence/gemma4_local_voice_v040.json`의 한국어 TTS→STT 정확 일치 PASS, `tests/test_local_voice.py`/API tests | 한 문장 smoke를 일반 WER로 확대하지 않고 다화자·잡음 corpus를 별도 품질 gate로 추가 |
| 106 | [x] | 음성→입력창 전사 | `COMPLETED` | `/api/workspace/voice/transcribe` 결과를 자동 전송 없이 편집 가능한 composer에 넣는 `app.js`, voice API/UI contract tests | transcript 편집과 명시적 전송 분리 유지 |
| 107 | [ ] | 녹음 시작·정지·취소 | `PARTIAL` | 30초/2 MiB 상한, start/stop/cancel/cleanup 상태 기계와 UI contract tests | 실제 브라우저에서 취소·권한 실패·최대시간·연속 녹음 E2E |
| 108 | [ ] | 음성 외부 전송 0 검증 | `PARTIAL` | 실제 voice evidence의 STT/TTS `external_calls=0`, 인증 loopback-only API와 fixed local runners | 완성 bundle에 대한 packet/egress 감사 |
| 109 | [x] | 선택적 로컬 TTS | `COMPLETED` | Windows System.Speech 고정 runner, 실제 Microsoft Heami ko-KR WAV smoke, 인증 API와 play/stop/object-URL cleanup tests | 설치 voice probe·사용자 실행·stop·크기/시간 상한과 외부 호출 0 유지 |
| 110 | [x] | 음성 미지원 이유 명시 | `COMPLETED` | capability payload와 disabled microphone tooltip/status | 지원 전까지 enabled 표시 금지 |

### K. 로컬 모델 선택 (111–117)

| ID | 체크 | 요구사항 | 상태 | 실제 근거 | 완료 승격 조건 |
|---:|:---:|---|---|---|---|
| 111 | [ ] | 로컬 모델 선택기 | `PARTIAL` | composer selector와 `/api/workspace/models/select`; 검증된 단일 모델만 표시 | 복수 manifest registry와 실제 worker 전환 |
| 112 | [x] | 현재 선택 Gemma 상태 표시 | `COMPLETED` | workspace capability/UI model selector tests | Fact-book와 실시간 일치 유지 |
| 113 | [ ] | 로컬 모델 자동 발견 | `NOT_IMPLEMENTED` | 고정 manifest 외 directory scan 없음 | 안전한 root, symlink/remote ID 차단, manifest 검증 |
| 114 | [ ] | 모델별 modality 지원 표시 | `PARTIAL` | metadata/Fact-book에는 modality 구성과 active 구분 | 복수 모델 registry 및 실제 processor gate |
| 115 | [ ] | 모델별 VRAM 요구량 표시 | `PARTIAL` | 목표/과거 실측 표시는 있으나 모델별 현재 실측 registry 없음 | 동일 장치·prompt scope의 current measurement |
| 116 | [ ] | 안전한 unload/load 전환 | `NOT_IMPLEMENTED` | 단일 worker lifecycle만 존재 | lease drain→unload→memory check→load→rollback E2E |
| 117 | [x] | 실제 모델과 목표 모델 구분 | `COMPLETED` | artifact/Fact-book/UI evidence rail | live verification 전 과거 모델 수치 재사용 금지 |

### L. 웹 검색·Lens.org (118–128)

| ID | 체크 | 요구사항 | 상태 | 실제 근거 | 완료 승격 조건 |
|---:|:---:|---|---|---|---|
| 118 | [x] | 웹 검색 기본 OFF | `COMPLETED` | `WebAccessPolicy`, UI capability state, tests | 모든 세션 시작을 offline으로 유지 |
| 119 | [ ] | 사용자 명시적 online 전환 | `PARTIAL` | 권한/allowlist 정책은 구현, 실제 executor 없음 | 세션 opt-in UI·감사 로그·즉시 revoke E2E |
| 120 | [ ] | 일반 웹 검색 connector | `NOT_IMPLEMENTED` | 정책과 UI 상태만 존재 | 승인 provider, bounded schema, URL/time provenance |
| 121 | [ ] | Lens 공식 API connector | `EXTERNAL_BLOCKER` | 고정 `api.lens.org` HTTPS POST connector·4중 gate·bounded schema/retry와 mocked transport tests 구현 | 승인 Bearer token·약관/플랜으로 실제 Lens 응답 검증 |
| 122 | [ ] | Lens 특허 검색 | `EXTERNAL_BLOCKER` | `/patent/search` 고정 executor·정규화·UI/API mocked E2E 구현 | 승인 token으로 official 특허 실응답 schema/attribution 검증 |
| 123 | [ ] | Lens 학술 검색 | `EXTERNAL_BLOCKER` | `/scholarly/search` 고정 executor·정규화·UI/API mocked E2E 구현 | 승인 token으로 official 학술 실응답 schema/attribution 검증 |
| 124 | [ ] | Lens ID/DOI/특허 링크 인용 | `PARTIAL` | Lens ID·identifier normalization과 `https://lens.org/<검증 ID>` allowlist, UI link/API tests | 실제 특허·학술 응답에서 DOI/특허 ID/링크 attribution E2E와 provenance drawer 연결 |
| 125 | [ ] | Lens 결과→AkasicDB 색인 | `PARTIAL` | `LensAkasicBridge`, normalized provenance 문서와 `/search-and-index` mocked E2E tests | 승인 token 실응답의 영속 graph/index·재시작·삭제 provenance E2E |
| 126 | [x] | 검색 query/domain/time 감사 provenance | `COMPLETED` | 고정 host/endpoint, retrieved time, query SHA-256, source-record SHA-256, Lens ID/canonical URL과 secret-redaction tests | token·원문 body 비노출, endpoint/time/query/source digest 계약 유지 |
| 127 | [x] | token 없으면 자동 비활성화 | `COMPLETED` | `WebAccessPolicy` capability state와 tests | 토큰 값을 UI/log/prompt에 노출하지 않기 |
| 128 | [x] | HTML scraping 금지 | `COMPLETED` | URL/endpoint 정책과 확장 로드맵 | 제품 데이터 경로를 공식 API로만 제한 |

### M. AI 워크스페이스 UI/UX (129–140)

| ID | 체크 | 요구사항 | 상태 | 실제 근거 | 완료 승격 조건 |
|---:|:---:|---|---|---|---|
| 129 | [x] | 채팅을 핵심 화면으로 확대 | `COMPLETED` | `cogni_demo/static/app.css`, 1080p UI tests/screenshots | 최신 EXE에서 시각 smoke |
| 130 | [x] | 입력창 sticky/항상 접근 가능 | `COMPLETED` | composer CSS/HTML과 `tests/test_demo_ui.py` | resize/zoom에서 가림 없음 확인 |
| 131 | [x] | 불필요한 스크롤 최소화 | `COMPLETED` | workspace grid/viewport CSS | 장문은 대화 pane만 스크롤 유지 |
| 132 | [x] | 첨부·RAG·모델·웹·음성을 composer에 배치 | `COMPLETED` | `static/index.html`, `app.js`, UI tests | 미구현 기능은 disabled 상태 유지 |
| 133 | [ ] | 근거 provenance drawer | `PARTIAL` | 메시지별 RAG source 표시만 구현 | 독립 drawer, 문장↔원문 위치 navigation |
| 134 | [x] | READY/THINKING/STREAMING/FAILED/COMPLETE 상태 | `COMPLETED` | `app.js`, agent API/UI tests | backend terminal state와 일치 유지 |
| 135 | [x] | 생성 취소 | `COMPLETED` | cancel API/UI, conversation transaction tests | cancel 후 worker drain/새 요청 정상 |
| 136 | [x] | 새 대화 | `COMPLETED` | new conversation UI/API tests | pending turn 중 안전 처리 유지 |
| 137 | [ ] | 반응형 layout | `PARTIAL` | breakpoint CSS와 1080p QA | 1366×768, 1920×1080, 4K, 125–200% zoom 시각/키보드 QA |
| 138 | [x] | 전역 가로 스크롤 금지 | `COMPLETED` | overflow CSS와 UI regression tests | 지원 해상도 matrix 유지 |
| 139 | [ ] | 접근성 | `PARTIAL` | ARIA/label/live region/disabled reason 일부 구현 | 키보드-only, focus order, screen-reader, contrast 감사 |
| 140 | [x] | active/gated/disabled 시각 구분 | `COMPLETED` | capability chips, disabled controls, UI tests | 실행 증거 없는 `ACTIVE/VERIFIED` 금지 |

### N. 좌측 메뉴·데모 페이지 (141–148)

| ID | 체크 | 요구사항 | 상태 | 실제 근거 | 완료 승격 조건 |
|---:|:---:|---|---|---|---|
| 141 | [x] | AI 워크스페이스 페이지 | `COMPLETED` | `static/index.html`, `app.js` | 최신 source smoke |
| 142 | [x] | 미션 컨트롤 | `COMPLETED` | dashboard markup/data와 UI tests | 실측/목표/계획 분류 유지 |
| 143 | [x] | 라이브 검증 | `COMPLETED` | validation page, server event stream | 현재 process 결과만 VERIFIED 승격 |
| 144 | [x] | 시스템 설계 | `COMPLETED` | architecture page와 `docs/ARCHITECTURE.md` | 실제 capability 상태와 일치 유지 |
| 145 | [x] | 사업 임팩트 | `COMPLETED` | business page와 `docs/COGNIBOARD_BUSINESS_DEMO_PLAN_KO.md` | 사업 목표를 실측으로 표시하지 않기 |
| 146 | [x] | 증빙·로드맵 | `COMPLETED` | evidence/roadmap page, Fact-book | evidence class·scope 표시 유지 |
| 147 | [x] | 사용자 매뉴얼/플레이북 | `COMPLETED` | `docs/COGNIBOARD_USER_MANUAL_PLAYBOOK_KO.md` | 신규 기능/버전에 맞춰 재생성 |
| 148 | [x] | 3분 IR 모드 | `COMPLETED` | IR UI flow와 UI tests | 자동 진행/중단/키보드 smoke |

### O. 검증·증거 무결성 (149–159)

| ID | 체크 | 요구사항 | 상태 | 실제 근거 | 완료 승격 조건 |
|---:|:---:|---|---|---|---|
| 149 | [x] | manifest 파일 무결성 | `COMPLETED` | `cogni_os/artifacts.py`, validation scripts/tests | load 전에 SHA 검증 유지 |
| 150 | [x] | 모델 SHA/parameter inventory | `COMPLETED` | artifact manifest/factbook/tests | 모델 변경 시 전부 무효화·재측정 |
| 151 | [x] | 현재 scope 실제 GPU 증거 | `COMPLETED` | 동결 commit `7039152`, E4B-it manifest/validator digest와 RTX 5090 Laptop actual run을 `validation/evidence/gemma4_cts_runtime_v040.json`에 결합 | 이 증거를 RTX 4090 또는 다른 commit/model 범위로 확대하지 않기 |
| 152 | [ ] | 현재 scope peak VRAM | `PARTIAL` | 과거 14.8469 GiB scoped canary | E4B-it 최신 source 및 target 4090에서 allocated/reserved 기록 |
| 153 | [x] | CTS depth/residual 현재 증거 | `COMPLETED` | 동결 commit `7039152` actual E4B-it: depth 100/100, 301 nodes, residual 0.0015335083, fallback/solver failure 0, finite PASS | source/model/manifest 변경 시 재실행 |
| 154 | [x] | fallback/solver failure telemetry | `COMPLETED` | runtime validator/evidence schema | silent fallback 금지 유지 |
| 155 | [x] | 전체 source 회귀 | `COMPLETED` | 동결 직전 `pytest` 891 passed, 6 skipped, 723 subtests; Ruff check/format, Node syntax, `git diff --check` PASS | 패키지 source에서도 핵심 gate 유지 |
| 156 | [x] | 과거 metric을 현재값으로 표시 금지 | `COMPLETED` | `cogni_os/factbook.py`, UI tests | scope 불일치 시 값 제거 |
| 157 | [x] | live 검증 전 `NOT VERIFIED` | `COMPLETED` | demo server/UI tests | startup 기본값 유지 |
| 158 | [x] | measured/verified/target/plan 분리 | `COMPLETED` | `cogni_os/evidence.py`, `config/evidence.schema.json` | 모든 신규 기능 동일 taxonomy 적용 |
| 159 | [ ] | 로그·시각·버전·commit 결합 | `PARTIAL` | evidence/log schema와 release records | 최신 bundle bytes까지 동일 commit/SHA로 묶기 |

### P. 문서·배포·사업 데모 (160–170)

| ID | 체크 | 요구사항 | 상태 | 실제 근거 | 완료 승격 조건 |
|---:|:---:|---|---|---|---|
| 160 | [x] | CogniBoard 사용자 매뉴얼 | `COMPLETED` | `docs/COGNIBOARD_USER_MANUAL_PLAYBOOK_KO.md` | 최신 UI 캡처/버전으로 갱신 |
| 161 | [x] | 운영 플레이북 | `COMPLETED` | 동일 문서의 실행·장애·검증 절차 | 신규 connector/voice 절차 추가 |
| 162 | [x] | 전체 아키텍처 구조도 | `COMPLETED` | `docs/ARCHITECTURE.md`, expansion roadmap Mermaid | 실제 데이터 흐름과 동기화 |
| 163 | [x] | Phase 1–11 계획·경계 | `COMPLETED` | `README.md`, Phase 8/9/10 문서, `docs/VALIDATION.md` | 상태 승격 시 evidence boundary 갱신 |
| 164 | [x] | 사업계획 연계 데모 계획 | `COMPLETED` | `docs/COGNIBOARD_BUSINESS_DEMO_PLAN_KO.md` | 사업 수치 출처와 plan 배지 유지 |
| 165 | [x] | 방산·신약·금융 데모 시나리오 | `COMPLETED` | business/demo UI·문서 | 실데이터/규제 검증 전 PoC로 표기 |
| 166 | [ ] | 현재 전체 소스 GitHub 업로드 | `PARTIAL` | 원격 브랜치는 `e090b3a`까지 push; 현재 변경/이 문서는 미push | clean commit push와 remote SHA 확인 |
| 167 | [ ] | 최신 current-source package | `NOT_IMPLEMENTED` | 기존 source ZIP은 최신 workspace 변경 전 | 동결 commit에서 wheel/source ZIP 재생성 |
| 168 | [ ] | 최신 version/checksum/release notes | `PARTIAL` | v0.4.0 version·release notes와 SBOM/third-party notice 생성 경로는 구현; 최종 동결 package 산출 전 | 동결 commit의 실제 bundle에서 `SHA256SUMS.txt`, SBOM/license/provenance를 생성·검증 |
| 169 | [ ] | 최신 더블클릭 데모 실행파일 | `NOT_IMPLEMENTED` | 기존 EXE는 최신 첨부/RAG/delete를 포함하지 않음 | 동일 동결 commit에서 EXE 생성·서명 상태 표기·smoke |
| 170 | [ ] | 재현 가능한 설치·실행 | `PARTIAL` | source 명령과 launcher 문서는 존재 | clean Windows 환경에서 offline 설치→manifest 검증→실행 재현 |

## 2. 미구현·부분·외부 차단 전용 체크리스트

아래 항목이 모두 `[x]`가 되기 전에는 “사용자 의도대로 모두 완성”이라고 보고하지 않는다.

### P0 — 현재 소스 무결성·RAG 영속성·배포

- [ ] **ID 8/9/166–170**: 동일 clean commit에서 source package, launcher/EXE, manual,
  release notes와 checksum을 만들고 패키지 E2E 후 GitHub remote SHA를 확인한다.
- [ ] **ID 152/159 (`PARTIAL`)**: 현재 장치 allocated VRAM 실측은 보존됐다. 목표 RTX 4090의
  allocated/reserved와 최종 bundle SHA를 동일 범위로 묶는다.

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
- [ ] **ID 111/114/115 (`PARTIAL`), 113/116 (`NOT_IMPLEMENTED`)**: 복수 manifest 자동 발견,
  modality/VRAM 상태, lease-safe unload/load/rollback을 구현한다.

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

현재 가장 먼저 닫아야 할 경로는 P0이다. P0가 끝나지 않은 상태에서 새 EXE를 만들면 오래된
기능을 다시 포장하게 된다. 멀티모달·음성은 입력 버튼보다 모델 tensor 계약과 VRAM gate가
선행되어야 한다. Lens는 connector 코드보다 승인 토큰·약관·감사 경계가 함께 준비되어야 한다.
Self-Harness 자동 승격은 OS 수준 격리와 rollback 증거보다 먼저 열 수 없다.

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
