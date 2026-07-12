# Cogni-OS 2.0 Genesis v0.3.0 릴리스 노트

## 핵심 결과

v0.3.0은 Phase 1~11의 안전한 소프트웨어 경로를 하나의 폐쇄망 제품 조립체로
연결한다. 로컬 Gemma 4 E4B 대화, 사실 기반 자기인식, 반복·중단 방어, causal
CTS/DEQ canary, typed 로컬 작업, 야간 FP-EWC/C-FIRE, bounded System 3/4, 연구 전용
AFlow, proposal-only Self-Harness를 포함한다.

Phase 1~11의 소프트웨어·안전 경계를 구현했고 현재-scope source regression,
자동화 20턴, 통합 GPU runtime, System 4 stress를 모두 통과했다. release artifact는
최종 소스에서 재생성했으며 byte digest는 `SHA256SUMS.txt`를 단일 기준으로 사용한다.

## 현재-scope 검증 요약

- 소스 회귀: Ruff check와 126-file format check, Node syntax check `PASS`;
  pytest 599 passed, 3 skipped, 3 deprecation warnings, 482 subtests passed
  (97.67초).
- 대화 완결성: 20/20턴 `PASS`; quality fallback 0, 문장 반복 0, cross-turn reuse 0,
  worker memory coverage 100%, Cogni-Core 생성 15턴, 생성 max/p95 78.516초,
  모든 턴 180초 이내, worker cleanup과 GPU lease release 완료.
- 통합 Gemma/DEQ/CTS: manifest 6 files, 모델
  `Gemma4ForConditionalGeneration`/hidden 2,560, depth 100/100, nodes 301/301,
  residual 0.0015945435, fallback/failure/failed edge 0, causal bridge non-zero,
  peak allocated VRAM 14.8468875885 GiB ≤ 16.7 GiB.
- System 4 CUDA 10,000회: p50/p95/p99/max
  2.9157/3.893715/5.217861/7.3089ms, convergence 1.0, settled FPR/FNR 0,
  topology switch 312/312.

GPU runtime 실측 장치는 NVIDIA GeForce RTX 5090 Laptop GPU다. 이 결과를 목표
RTX 4090 24GB 인증으로 간주하지 않으며, System 4 결과도
measurement-only/advisory다. 대화 시험의 per-process GPU memory는 driver 미보고로
별도 `unverified`다.

## 제품 권한

- 로컬 manifest-verified Gemma: `authoritative`
- causal CTS/DEQ: `canary`
- System 1.5: `gated`
- System 2.5: `night_only`
- BIO-HAMA/System 3/System 4: `advisory`
- AFlow: `research_archive_only`
- Self-Harness: `proposal_only`

Self-Harness는 증거와 후보를 저장하지만 active source를 실행·덮어쓰기·승격하지
않는다. 자동 승격에는 v0.3.0 범위 밖의 Phase 12 kernel-isolated attestation과
fault-injection 증거가 필요하다.

Gemma base-only/last-known-good 생성 fallback은 제품 factory에서 비활성화된다.
Cogni-Core terminal 성공과 base integrity 확인 전에는 답변을 publish하지 않고,
Core 실패는 fail-closed로 끝난다. 1회의 bounded 응답 복구 뒤 사용할 수 있는 정적
안전 응답은 Gemma fallback이 아니다.

## 주요 변경

- model/code/config/device scope에 묶인 content-addressed evidence와 Fact-book LKG;
- 거짓 7B 자기소개, 반복 loop, 역할 토큰, 문장 중단을 commit 전에 차단;
- lease/job/deadline/artifact/session-bound Model IPC v3;
- fixed 301-node CTS V2와 bounded causal logits conditioner;
- trained-artifact-gated Fast Weight와 System 3 독립 authority gate;
- exact 28-agent System 4 topology/session/PCAS stress telemetry;
- immutable TypedTaskPlan, one-use capability, T0 read/output artifact write,
  제품 기본 차단인 T1 fixed-test trusted opt-in과 T2 proposal staging;
- sealed AFlow research archive와 K≥3 Self-Harness proposal ledger;
- bounded journal에서 성공·실패·proposal·negative record를 엄격히 검증해 복원하는
  Self-Harness restart hydration;
- v0.3.0 version metadata, launcher preflight, 시작 시 `UNVERIFIED`이며 현재 성공한
  실행만 증거로 승격하는 UI capability 표시.

## 실행파일 주의

`CogniBoard-v0.3.0.exe`는 console-free launcher이지 standalone model bundle이나
서명 installer가 아니다. 동일 source tree, 로컬 dependency, 모델과 manifest가
필요하다. 최종 byte에 대한 checksum은 모든 검증 후 `SHA256SUMS.txt`에 생성한다.
v0.3.0 EXE, wheel과 source archive는 최종 소스에서 재생성했다. 같은 디렉터리의
`SHA256SUMS.txt`로 byte 동일성을 확인해야 하며, 이 전의 파일은 서명 installer나
최종 배포 승인 근거가 아니다.

## 아직 완료로 주장하지 않는 항목

- RTX 4090 24GB 동일 artifact/config 반복 실측과 packet/egress audit;
- 독립 외부 human-labelled 20턴 대화와 false-complete/repetition/중단 판정;
- 학습된 DEQ/Wproj, Fast Weight Programmer와 System 3 expert artifact;
- 독립 held-out 품질과 최소 3-seed FP-EWC BWT/FWT;
- production PCAS calibration corpus와 System 4 quality ablation;
- 실제 production failure 99% capture와 95% cluster precision;
- attested sandbox 기반 safe promotion;
- 코드 서명, 서명 installer/update trust chain, SBOM·license·provenance bundle;
- AGI, 무한 자기개선, O(1) total memory.

전체 재현 명령과 최신 개발 canary는
[`docs/VALIDATION.md`](../docs/VALIDATION.md),
[`docs/GEMMA4_VALIDATION.md`](../docs/GEMMA4_VALIDATION.md),
[`COGNI_OS_2_VALIDATION_REPORT_KO.md`](COGNI_OS_2_VALIDATION_REPORT_KO.md)를
참고한다.
