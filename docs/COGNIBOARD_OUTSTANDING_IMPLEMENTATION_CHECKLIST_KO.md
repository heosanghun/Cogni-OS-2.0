# CogniBoard 미완료 항목 전용 체크리스트

> 이 문서는 `COGNIBOARD_MASTER_ACCEPTANCE_CHECKLIST_KO.md`에서 자동 생성됩니다.
> 직접 수정하지 말고 마스터 원장의 상태·근거·승격 조건을 먼저 갱신하십시오.

## 현재 집계

- 전체 미완료: **55개**
- 코드/제품 경로 미구현: **9개**
- 부분 구현 또는 검증 잔여: **41개**
- 외부 장치·토큰·아티팩트 차단: **5개**

## 제품 실행 경로 미구현

| ID | 체크 | 요구사항 | 현재 근거 | 완료 승격 조건 |
|---:|:---:|---|---|---|
| 69 | [ ] | 격리된 실제 patch 실행 | v0.3.2는 `proposal_only`; `README.md` | Phase 12 kernel/process/network 격리 sandbox |
| 72 | [ ] | 검증 후 승격/승인 | 자동 승격은 명시적으로 차단 | 사람 승인 + attested promotion transaction |
| 73 | [ ] | 기존 코드 rollback | 후보 lifecycle rollback은 source rollback이 아님 | 실제 설치 bytes의 원자적·byte-identical rollback |
| 75 | [ ] | UI가 아닌 실제 자가수정 E2E | Self-Harness UI는 proposal-only 상태 표시 | 실패 재현→patch→격리검증→승인→승격→rollback E2E |
| 85 | [ ] | 비디오 processor 연결 | video token/config 표시는 실행 권한이 아님 | frame/sampling/time/VRAM 상한과 실제 추론 |
| 100 | [ ] | 검증된 로컬 semantic embedder | 현재 검색은 안정적 lexical projection | 모델 artifact/manifest, 품질·VRAM·라이선스 검증 |
| 113 | [ ] | 로컬 모델 자동 발견 | 고정 manifest 외 directory scan 없음 | 안전한 root, symlink/remote ID 차단, manifest 검증 |
| 116 | [ ] | 안전한 unload/load 전환 | 단일 worker lifecycle만 존재 | lease drain→unload→memory check→load→rollback E2E |
| 120 | [ ] | 일반 웹 검색 connector | 정책과 UI 상태만 존재 | 승인 provider, bounded schema, URL/time provenance |

## 부분 구현·검증 잔여

| ID | 체크 | 요구사항 | 현재 근거 | 완료 승격 조건 |
|---:|:---:|---|---|---|
| 6 | [ ] | 기본 air-gap 및 외부 호출 0 | `cogni_os/runtime.py`, `cogni_demo/server.py`, `tests/test_airgap.py` | 현재 배포 바이트에 대한 패킷/egress 감사와 오프라인 의존성 검토 |
| 10 | [ ] | Gemma 특징과 DEQ 고정점 경로 결합 | `cogni_agent/core_pipeline.py`, `cogni_core/deq.py`, `tests/test_core_pipeline.py` | 학습된 DEQ/Wproj와 held-out 인과 ablation으로 canary 해제 |
| 14 | [ ] | 추론 깊이에 선형 증가하지 않는 메모리 | CTS arena와 solver history는 고정; `docs/VALIDATION.md` | 모델·로그·외부 데이터 전체 O(1) 주장은 금지하고 심도별 실제 곡선 재측정 |
| 19 | [ ] | System 1.5 Fast Weight | `cogni_core/fast_weights.py`, `tests/test_fastweight_runtime.py`; Fact-book `gated` | 승인된 학습 checkpoint와 AQ/OOD·비회귀 증거 |
| 20 | [ ] | DEQ 상태를 저랭크 임시 가중치로 변환 | `cogni_core/fast_weights.py`, `cogni_core/adaptation.py` | 실제 학습 artifact로 answer-quality 이득 검증 |
| 22 | [ ] | System 2.5 FP-EWC | `cogni_core/fp_ewc.py`, `tests/test_fp_ewc.py`; `night_only` | 실제 도메인 3-seed BWT/FWT 증거 |
| 23 | [ ] | Fisher 기반 망각 방지 | empirical Fisher/anchor/merge 구현과 `tests/test_fp_ewc.py` | 독립 데이터셋의 망각 감소와 품질 비회귀 |
| 24 | [ ] | C-FIRE spectral norm < 0.95 | `cogni_core/c_fire.py`, `tests/test_c_fire.py` | 전체 적용 대상과 합성 연산의 인증서·실측; 전체 decoder 수축 주장 금지 |
| 25 | [ ] | System 3 Sparse MoE | `cogni_core/experts.py`, `cogni_core/expert_lifecycle.py`, Phase 8 tests | 학습·교정 expert checkpoint와 독립 verifier |
| 26 | [ ] | novelty 기반 routing | `cogni_core/routing.py`, `tests/test_routing.py` | 실제 OOD corpus threshold calibration |
| 27 | [ ] | 안전한 expert spawn·등록·rollback | `cogni_core/expert_lifecycle.py`, `tests/test_expert_lifecycle.py` | 제품 artifact promotion의 독립 검증 |
| 28 | [ ] | System 4 Tensor Swarm | `cogni_core/swarm.py`, `cogni_core/swarm_sessions.py`, `tests/test_swarm.py`; `advisory` | production PCAS corpus와 답변 품질 ablation |
| 29 | [ ] | 내부 에이전트 tensor 중심 통신 | System 4 tensor 경로와 `cogni_os/tensor_service.py`; 제어/제품 전체가 tensor-only는 아님 | 모든 대상 경계의 명시적 tensor schema·복사/직렬화 측정 |
| 31 | [ ] | BIO-HAMA meta-router | `cogni_core/meta_router.py`, `tests/test_meta_router.py`; `advisory` | 실제 routing quality calibration |
| 32 | [ ] | ADAS/AFlow pipeline 생성·탐색 | `cogni_flow/aflow.py`, `cogni_flow/aflow_research.py`; `research_archive_only` | attested evaluator와 제품 설치 권한 분리 검증 |
| 39 | [ ] | 임의의 일반 질문 응답 | 실제 Gemma 대화 경로 존재; typed tool 권한은 제한 | 독립 human-labelled 다양한 주제 acceptance |
| 54 | [ ] | 코드 작성·수정 | output-only write와 T2 proposal staging; 실제 source mutation 금지 | 승인·격리·회귀·rollback이 결합된 source patch 경로 |
| 55 | [ ] | PoC/MVP 개발 지원 | `cogni_agent/tools.py`의 bounded `/project` 다중 파일 묶음, `cogni_flow/task_plan.py`, `tests/test_agent_project_bundle.py`; output-only·무실행·무덮어쓰기 | 자연어 요청→검토 가능한 typed bundle 생성과 선택적 격리 테스트 E2E |
| 56 | [ ] | 테스트 실행 | 고정 pytest primitive, `cogni_flow/task_plan.py`; 기본 차단 | OS 격리와 명시적 trusted opt-in |
| 61 | [ ] | 임의 shell·네트워크·경로 탈출 차단 | 정책/경로 단위 테스트는 통과; `docs/SECURITY.md` | process-tree와 network를 강제하는 OS sandbox 실증 |
| 63 | [ ] | 적용 전 회귀 테스트 | 후보 평가/negative archive는 구현 | source patch 격리 실행과 전체 gate 연결 |
| 64 | [ ] | 취소·timeout·rollback | IPC/task deadline·취소·candidate rollback tests | 실제 source/bundle 변경의 byte-identical rollback |
| 70 | [ ] | 후보 회귀·보안 테스트 | sealed evaluator/negative archive 단위 경로 | 실제 patch sandbox에서 전체 회귀·fault injection |
| 78 | [ ] | PDF 텍스트 추출·색인 | 로컬 `pypdf` strict 추출을 Windows Job Object/POSIX rlimit의 256MiB·CPU 6초·wall 8초 격리 worker에서 실행; 128쪽/문자 상한·preview·AkasicDB 색인·재색인과 tests 구현 | PDF page 번호 provenance와 악성 PDF corpus 추가 |
| 86 | [ ] | 멀티모달 VRAM 경계 검증 | 이미지와 오디오 actual-model smoke는 PASS이나 증거 JSON에 peak allocated/reserved VRAM은 없음 | image/audio 조합을 current commit에서 16.7 GiB/finite/latency로 실측 |
| 102 | [ ] | 답변 provenance drawer | 파일명·chunk·score 근거 표시와 첨부 preview 경로 | 문장별 클릭, PDF page/원문 위치 navigation, raw/summary 구분 |
| 103 | [ ] | 마이크 입력 | 브라우저 `getUserMedia`→16 kHz mono WAV→인증 loopback STT 경로와 UI/API contract tests | 실제 Windows 마이크 장치에서 권한·녹음·전사 브라우저 E2E |
| 104 | [ ] | Windows 마이크 권한 처리 | 클릭 시에만 권한 요청하고 오류/취소 cleanup UI 경로 구현 | 권한 거부·철회·장치 없음·장치 전환을 실제 브라우저에서 검증 |
| 107 | [ ] | 녹음 시작·정지·취소 | 30초/2 MiB 상한, start/stop/cancel/cleanup 상태 기계와 UI contract tests | 실제 브라우저에서 취소·권한 실패·최대시간·연속 녹음 E2E |
| 108 | [ ] | 음성 외부 전송 0 검증 | 실제 voice evidence의 STT/TTS `external_calls=0`, 인증 loopback-only API와 fixed local runners | 완성 bundle에 대한 packet/egress 감사 |
| 111 | [ ] | 로컬 모델 선택기 | composer selector와 `/api/workspace/models/select`; 검증된 단일 모델만 표시 | 복수 manifest registry와 실제 worker 전환 |
| 114 | [ ] | 모델별 modality 지원 표시 | metadata/Fact-book에는 modality 구성과 active 구분 | 복수 모델 registry 및 실제 processor gate |
| 115 | [ ] | 모델별 VRAM 요구량 표시 | 목표/과거 실측 표시는 있으나 모델별 현재 실측 registry 없음 | 동일 장치·prompt scope의 current measurement |
| 119 | [ ] | 사용자 명시적 online 전환 | 권한/allowlist 정책은 구현, 실제 executor 없음 | 세션 opt-in UI·감사 로그·즉시 revoke E2E |
| 124 | [ ] | Lens ID/DOI/특허 링크 인용 | Lens ID·identifier normalization과 `https://lens.org/<검증 ID>` allowlist, UI link/API tests | 실제 특허·학술 응답에서 DOI/특허 ID/링크 attribution E2E와 provenance drawer 연결 |
| 125 | [ ] | Lens 결과→AkasicDB 색인 | `LensAkasicBridge`, normalized provenance 문서와 `/search-and-index` mocked E2E tests | 승인 token 실응답의 영속 graph/index·재시작·삭제 provenance E2E |
| 133 | [ ] | 근거 provenance drawer | 메시지별 RAG source 표시만 구현 | 독립 drawer, 문장↔원문 위치 navigation |
| 137 | [ ] | 반응형 layout | breakpoint CSS와 1080p QA | 1366×768, 1920×1080, 4K, 125–200% zoom 시각/키보드 QA |
| 139 | [ ] | 접근성 | ARIA/label/live region/disabled reason 일부 구현 | 키보드-only, focus order, screen-reader, contrast 감사 |
| 152 | [ ] | 현재 scope peak VRAM | 과거 14.8469 GiB scoped canary | E4B-it 최신 source 및 target 4090에서 allocated/reserved 기록 |
| 170 | [ ] | 재현 가능한 설치·실행 | source 명령과 launcher 문서는 존재 | clean Windows 환경에서 offline 설치→manifest 검증→실행 재현 |

## 외부 입력 없이는 완료 불가능

| ID | 체크 | 요구사항 | 현재 근거 | 완료 승격 조건 |
|---:|:---:|---|---|---|
| 4 | [ ] | RTX 4090 24GB 목표 장치 검증 | 목표는 `config/default.toml`; 기존 실측은 `release/evidence/phase11_gemma_cts_runtime_20260712.json`의 RTX 5090 Laptop | 정확한 RTX 4090에서 동결 소스·E4B-it 전체 게이트 재실측 |
| 16 | [ ] | 16.7 GiB VRAM 방어선 | `scripts/validate_gemma4_runtime.py` postcondition; 과거 5090 실측 14.8469 GiB | 목표 RTX 4090·E4B-it·현재 소스에서 allocated/reserved 모두 재검증 |
| 121 | [ ] | Lens 공식 API connector | 고정 `api.lens.org` HTTPS POST connector·4중 gate·bounded schema/retry와 mocked transport tests 구현 | 승인 Bearer token·약관/플랜으로 실제 Lens 응답 검증 |
| 122 | [ ] | Lens 특허 검색 | `/patent/search` 고정 executor·정규화·UI/API mocked E2E 구현 | 승인 token으로 official 특허 실응답 schema/attribution 검증 |
| 123 | [ ] | Lens 학술 검색 | `/scholarly/search` 고정 executor·정규화·UI/API mocked E2E 구현 | 승인 token으로 official 학술 실응답 schema/attribution 검증 |

## 완료 판정 규칙

- 코드 파일이나 버튼의 존재만으로 완료 처리하지 않습니다.
- 제품 경로 연결, bounded/fail-closed 안전성, 자동 회귀 또는 실측 증거가 모두 있어야 합니다.
- RTX 4090과 승인된 Lens 토큰·약관처럼 외부 입력이 필요한 항목은 제공 전까지 차단 상태를 유지합니다.
- 완료된 ID는 이 문서에서 자동으로 사라지고 마스터 원장에만 `[x]`로 남습니다.
