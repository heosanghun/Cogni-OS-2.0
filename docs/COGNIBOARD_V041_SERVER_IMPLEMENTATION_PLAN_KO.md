# CogniBoard v0.4.1 서버 구현 및 검증 계획

작성일: 2026-07-17 KST
작업 브랜치: `server/evidence-spine-v0.4.1`
서버 작업공간: `/home/shoon/workspace/Cogni-OS-2.0-v041`
기준 소스: v0.4.0 `5efce859e8f3f7793ad133f26f2c4390001d63d4`
Linux 기준선 수정 커밋: `9d1f5b9`

## 1. 목표와 현재 진실

v0.4.1의 직접 목표는 로컬 RAG 답변이 어떤 파일의 어느 페이지와 청크를 사용했는지 재현 가능하게 증명하는 Evidence Spine을 완성하고, Linux 서버에서도 동일한 런타임 계약을 지키는 것이다.

현재 CPU 기준선은 `888 passed, 9 skipped, 0 failed`이다. GPU를 노출하지 않고 네트워크를 차단한 컨테이너에서 얻었다. 기존 FP-EWC 별칭 경고 3건은 추적 대상으로 남긴다. 릴리스 원시 로그와 이미지 digest 결합은 Phase 11에서 새로 생성한다.

현재 정량 진행률은 15%다. 코드를 작성한 것만으로는 완료되지 않으며 Exit Gate를 현재 커밋에서 통과해야만 진행률을 획득한다.

## 2. 절대 GPU 규칙

이 절은 권고가 아니라 MUST 및 ABORT 규칙이다.

- 연구실 전체 허용 범위는 물리 GPU 0~5다. 이것은 개별 프로젝트가 0~5를 모두 사용할 수 있다는 뜻이 아니다.
- 이 프로젝트는 물리 GPU 5 하나만 사용할 수 있다.
- 물리 GPU 5 UUID는 `GPU-84d7eeb0-65e0-a5b1-d7db-d09ef59fe03a`다.
- 물리 GPU 0~4는 연구실 차원에서만 허용된 장치다. 이 프로젝트는 0~4를 조회·열거·노출·예약·할당·사용하지 않으며 기존 프로세스, 컨테이너, tmux, CUDA context도 시작·중지·변경하지 않는다.
- 물리 GPU 6·7은 연구실 금지 장치다. 모든 단계에서 조회·열거·노출·예약·할당·사용하지 않는다.
- 프로젝트 GPU 경계 밖인 0~4와 6·7에 대한 위 동작은 모두 0회여야 한다.
- CPU 게이트를 모두 통과하기 전에는 GPU 작업을 시작하지 않는다.
- 호스트 네이티브 작업은 `CUDA_VISIBLE_DEVICES=5`를 사용한다.
- Docker 작업은 `--gpus device=5 --network none`을 사용한다.
- Docker 내부에서는 remap된 장치가 일반적으로 논리 `cuda:0`이다. 내부에 다시 `CUDA_VISIBLE_DEVICES=5`를 설정해 유일한 장치를 숨기지 않는다.
- 실행 직전과 직후 호스트에서 `nvidia-smi -i 5`만 사용해 index, UUID, PID, utilization, memory를 기록한다.
- GPU 5에 외부 PID가 있거나 UUID가 다르거나 하나 이외의 CUDA 장치가 보이거나 사전 상태가 idle이 아니면 ABORT한다.
- 서버 RTX A6000 결과를 RTX 4090 결과로 표기하지 않는다. RTX 4090 24GB 실기기 검증은 `EXTERNAL_BLOCKER`로 유지한다.

## 3. 완료 증거 스키마

모든 VERIFIED 증거는 `release/evidence/v0.4.1` 아래 content-addressed JSON과 원시 로그로 저장한다.

필수 필드:

- source commit과 clean 또는 dirty tree digest
- container image digest와 Python·PyTorch·Transformers·PDF 라이브러리 버전
- model manifest와 각 모델 파일 digest
- 물리 GPU index·UUID·driver와 컨테이너 논리 index
- 정확한 command·argv·network mode·UTC/KST 시작과 종료·exit code
- 수집·통과·건너뜀·실패 테스트 수
- stdout·stderr SHA-256와 입력·악성 corpus digest
- peak allocated·reserved VRAM, latency, CTS depth, residual, fallback, finite
- GPU 5 사전·사후 snapshot digest

`VERIFIED`는 validator exit 0과 모든 scope hash 일치 때만 표시한다. `COMPLETED`는 acceptance ID가 증거 경로와 digest를 가지고 독립 verifier가 PASS할 때만 허용한다. partial, not_implemented, external_blocker는 그대로 공개한다.

## 4. Phase 1~11 권한과 게이트

| Phase | 권한 | v0.4.1 영향 | 완료 게이트 | 가중치 |
|---|---|---|---|---:|
| 1. 대화 무결성·근거화 | direct | RAG 요청 보존, 0-hit, citation | 반복·중단·무근거 우회 0 | 15% |
| 2. 생명주기·GPU lease·IPC | direct | Linux retirement ACK, GPU5 격리 | 변조 경로 무토큰 공개, CPU 회귀 PASS | 10% |
| 3. Gemma-DEQ canary | regression | causal binding 회귀 | 실제 모델 DEQ 사용과 base-only 대조 | 8% |
| 4. CTS V2 canary | direct | deterministic checkpoint, depth 100 | 재현, finite, fallback 0 | 8% |
| 5. System 1.5 | gated | 회귀만 | 학습 checkpoint 없이는 승격 금지 | 8% |
| 6. System 2.5 | night_only | C-FIRE·FP-EWC 회귀 | 주·야간 배제, 3-seed 증거 별도 | 8% |
| 7. System 4 | advisory | tensor-only 회귀 | 텍스트 IPC 0, advisory 유지 | 7% |
| 8. System 3 | advisory | sparse expert 회귀 | 무단 spawn·가중치 변경 0 | 7% |
| 9. 로컬 작업·첨부·RAG source | direct | PDF provenance, source API, Drawer | locator·digest·원문 100% 일치 | 12% |
| 10. AFlow | research_archive_only | 격리 회귀 | 설치·실행 경로 0 | 5% |
| 11. Self-Harness·릴리스 | proposal_only | ledger, 패키지, 매뉴얼 | active mutation 0, digest 결합 | 12% |

v0.4.1은 Phase 3·5·6·7·8·10·11의 권한을 승격하지 않는다. 표시 권한은 Runtime Fact-book의 실제 상태만 따른다.

## 5. 구현 순서와 Exit Gate

### Stage A — Linux 기준선 고정 완료

- worker retirement ACK로 Linux tensor resource-sharer 수명 경쟁 제거
- CTS evidence를 고정 순서 scalar 계산으로 재현하고 checkpoint digest 갱신
- straight-through mask forward를 정확한 hard k-hot으로 고정
- 집중 회귀 `25 passed, 1 skipped`
- 전체 회귀 `888 passed, 9 skipped, 0 failed`
- Ruff check·format과 `git diff --check` exit 0
- CPU only, `--network none`, GPU mount 0

### Stage B — PDF Evidence Schema v1

- 빈 페이지를 포함한 1-based 물리 페이지 순서 보존
- 페이지별 청크화로 페이지 경계 횡단 금지
- `page_number`, `char_start`, `char_end`, `offset_basis`, `excerpt_sha256` 저장
- index, rebuild, reindex, rollback 경로에 동일 typed record 사용
- `page_text[char_start:char_end] == chunk_text` 100%
- `sha256(chunk_text) == excerpt_sha256` 100%
- 빈 2페이지 뒤 3페이지가 page 3으로 유지
- 재시작·reindex 후 locator 동일

### Stage C — RAG 요청 계약

- evidence와 별개인 `retrieval_requested` boolean 전달
- RAG-on 0-hit에서 모델·Fast Path·Fact-book으로 조용히 우회하지 않음
- bounded 무근거 응답과 citation 0개 반환
- 0-hit model start 0회, evidence 시 Fast Path·Fact-book 0회
- RAG-off 기존 대화 회귀 PASS

### Stage D — Exact Source API

- `GET /api/workspace/rag/source?attachment_id=...&chunk_index=...`
- 인덱스 relational record의 정확한 청크만 반환
- 인증, query key, 중복 값, ID·범위, path leakage를 fail-closed 처리
- 검색 결과와 API의 text·locator·digest 완전 일치
- 추가·중복 query, 음수·128 이상 index는 4xx
- host·project path 노출 0

### Stage E — Evidence Drawer

- 답변 `[근거 N]`을 DOM text node와 button으로 안전 변환
- 우측 Drawer에 제목, 위치, score, digest, exact excerpt 표시
- Escape, backdrop close, focus trap, focus return, request race 차단
- 620px 이하 full-width sheet
- `innerHTML`과 script injection 0
- 키보드 open·read·close 가능
- source digest 불일치 시 원문 표시 금지

### Stage F — 악성 corpus와 품질 회귀

- 빈·암호화·손상·대형·다중 페이지 PDF
- RAG positive·0-hit·duplicate·rollback
- API 인증·query pollution·path leakage
- 반복 답변·역할 토큰·미완 문장·거짓 identity
- JS syntax·XSS·접근성·bounded DOM

Exit Gate는 focused tests, 전체 pytest, Ruff check, Ruff format check, Node syntax, `git diff --check`가 모두 exit 0인 것이다. 신규 테스트 추가 뒤 통과 수는 888보다 작을 수 없고 skip 변화는 사유를 기록한다.

### Stage G — 물리 GPU 5 실제 모델 검증

CPU 게이트가 모두 녹색인 같은 커밋에서만 수행한다. GPU5 preflight 뒤 Docker에 물리 GPU5 하나만 노출한다. 모델과 manifest는 read-only, network는 none이다.

- 모든 Stage G 실행은 `scripts/gpu5_boundary_guard.py` 단일 진입점을 거치며 직접 Docker 또는 validator 실행은 금지한다.
- 호스트 GPU 질의는 반드시 `nvidia-smi -i 5` 형태만 허용하고 selector 없는 전 장치 질의는 실행 전에 거부한다.
- 컨테이너 이미지는 `cogni-os-dev@sha256:20aaf1d7cde8d6a504ba08f158a34a1907eac9413f3578acc4637f0a1b2ec8ba`로 고정한다.
- Docker 옵션은 `--gpus device=5 --network none --log-driver none`과 고유한 `--name`만 허용한다.
- 컨테이너 환경 변수는 `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `PYTHONDONTWRITEBYTECODE=1`만 허용하고 CUDA·PYTHONPATH·LD_PRELOAD override는 거부한다.
- 정확한 두 read-only mount만 허용한다: 저장소 `/home/shoon/workspace/Cogni-OS-2.0-v041:/workspace:ro`, 모델 `/home/shoon/models/gemma4-e4b:/models/gemma4-e4b:ro`.
- workdir은 정확히 `/workspace`이며 추가·상위·형제·부분 subtree mount, read-write mount, privilege 상승을 거부한다.
- 컨테이너 CUDA 장치는 정확히 1개여야 하고 논리 index는 `cuda:0`, 물리 UUID는 고정 GPU5 UUID와 일치해야 한다.
- validator script와 option은 고정 allowlist, 개수·문자 길이·수치 범위를 적용하며 unknown·duplicate·shell token·내부 output 경로는 실행 전에 거부한다.
- 증거 파일은 컨테이너에 mount하지 않는다. 호스트가 신규 파일로만 생성하고 4MiB로 제한하며 stdout·stderr와 SHA-256을 보존한다.
- timeout, 예외, non-zero 종료에서도 고유 이름 컨테이너를 stop·remove한 뒤 부재와 GPU5 idle을 postflight로 확인한다.
- 물리 GPU 0~4와 6·7은 조회·열거·노출·예약·할당·사용 모두 0회
- manifest closed-world PASS
- 대화·RAG·PDF citation·cancellation·mutation tripwire PASS
- GPU 상주 turn의 100%에서 고정 GPU5 UUID aggregate memory를 샘플링하고 모든 sample이 16.7GiB 이하, overlimit 0
- peak VRAM 16.7GiB 이하, residual 충족, fallback 0, finite true
- 사후 GPU memory와 process가 기준 상태로 복귀

### Stage H — Phase 11 인수·패키지·문서

- acceptance와 outstanding ledger 갱신
- CogniBoard 플레이북과 메뉴 설명 갱신
- package EXE SHA를 source commit과 smoke evidence SHA에 결합
- 검증된 커밋과 증거만 GitHub push
- master validator PASS
- 미구현·외부 차단을 완료로 표시한 항목 0
- stale evidence VERIFIED 경로 0

## 6. 즉시 중단 조건

다음 하나라도 발생하면 작업을 중단하고 VERIFIED와 COMPLETED 승격을 금지한다.

- CPU 회귀 실패 또는 설명 없는 test count 감소
- dirty 또는 증거와 결합되지 않은 source
- source API의 경로·token·내부 예외 유출
- model·manifest·checkpoint·excerpt digest 불일치
- 비인가 network 요청
- GPU5 UUID 불일치, 외부 PID, 비-idle 상태
- 물리 GPU5 이외 장치 조회·열거·노출·예약·할당·사용
- peak allocated VRAM 16.7GiB 초과
- non-finite, residual 초과, fallback 사용
- active source mutation 또는 증거 없는 자동 patch

## 7. 현재 진행률

| 구간 | 상태 | 인정 진행률 |
|---|---|---:|
| Stage A Linux 기준선 | 완료 | 15% |
| Stage B PDF provenance | 진행 중 | 0% |
| Stage C RAG 요청 계약 | 진행 중 | 0% |
| Stage D Exact Source API | 대기 | 0% |
| Stage E Evidence Drawer | 대기 | 0% |
| Stage F 악성 corpus | 대기 | 0% |
| Stage G GPU5 검증 | 대기 | 0% |
| Stage H 인수·패키지 | 대기 | 0% |
| 합계 | 진행 중 | 15% |

다음 진행률 증가는 Stage B 또는 C의 Exit Gate를 현재 커밋에서 완전히 통과한 뒤에만 기록한다.
