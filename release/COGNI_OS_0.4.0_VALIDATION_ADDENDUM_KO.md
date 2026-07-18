# Cogni-OS 2.0 Genesis v0.4.0 검증 부록

## 증거 원칙

단위 테스트 통과는 software invariant 증거다. model 품질, 목표 GPU 메모리,
폐쇄망 배포, retrieval 정확도, 음성·시각 품질 또는 안전한 자동 source promotion을
자동으로 증명하지 않는다. 모든 측정은 exact model, manifest, source commit,
configuration, device scope에 묶여야 한다.

## 실제 로컬 음성 smoke

2026-07-16 개발 장치에서 다음 end-to-end 경로를 실행했다.
보존된 raw JSON은 `validation/evidence/gemma4_local_voice_v040.json`이다.

1. 설치된 Microsoft Heami Desktop `ko-KR` Windows voice로 한 문장을 합성했다.
2. 합성 WAV를 16 kHz mono 16-bit PCM으로 bounded resample했다.
3. seven-file manifest로 검증된 local `gemma4-e4b-it` worker를 로드했다.
4. manifest-bound `Gemma4Processor` audio chat template와 고정 tensor IPC를
   통과시켰다.
5. resident Gemma가 전사한 문장과 입력 문장을 정규화해 비교했다.

| 필드 | 관측값 |
|---|---|
| 입력 | `안녕하세요. 코그니보드 로컬 음성 검증입니다.` |
| 전사 | `안녕하세요. 코그니보드 로컬 음성 검증입니다.` |
| normalized similarity | `1.0` (gate `≥ 0.70`) |
| admitted duration | `5.0187 seconds` |
| admitted audio | `16 kHz`, mono, 16-bit PCM WAV |
| TTS voice | `Microsoft Heami Desktop`, `ko-KR` |
| STT/TTS application external calls | `0 / 0` |

이 명령은 현재 배포하지 않는다. 기록은 개발 호스트의 historical observation이며,
voice validator가 E4B-it 전용 GPU5 guard allowlist와 실행 전후 identity 계약에
들어오기 전까지 current-scope 재현은 `BLOCKED`다.

이 결과는 단일 synthetic TTS phrase의 pipeline smoke다. multi-speaker WER,
noise robustness, far-field microphone, multilingual speech, streaming latency,
TTS naturalness 또는 accessibility acceptance evidence가 아니다.

## 실제 로컬 image smoke

2026-07-16 개발 장치에서 locally generated 256×256 PNG의 중앙에 하나의
파란색 정사각형을 만들고 manifest-bound local `gemma4-e4b-it` Cogni-Core
image route에 전달했다. 보존된 raw JSON은
`validation/evidence/gemma4_local_image_v040.json`이다.

| 필드 | 관측값 |
|---|---|
| PNG byte count | `595` |
| PNG SHA-256 | `0e513cbdb6792a57d4b16db147085d043ee344622e19b3527c4d79093a390848` |
| 응답 | `중앙의 큰 도형은 파란색 정사각형입니다.` |
| required concept gate | `blue=true`, `square=true` |
| finish / generation mode | `stop` / `cogni_core` |
| application external calls | `0` |

이 명령은 현재 배포하지 않는다. 기록은 개발 호스트의 historical observation이며,
image validator가 E4B-it 전용 GPU5 guard allowlist와 실행 전후 identity 계약에
들어오기 전까지 current-scope 재현은 `BLOCKED`다.

이 결과는 고정된 blue-square 한 장에 대한 smoke다. general visual reasoning,
OCR, document image understanding, adversarial image safety 또는 combined
multimodal depth-100 VRAM evidence가 아니다.

## v0.4.0 신규 deterministic gate

- attachment catalog/blob restart hydration, digest mismatch, capacity, deletion,
  preview, isolated PDF process wall/CPU/RAM and page/character bounds, and
  reindex transaction;
- pinned AkasicDB revision/module digests, bounded document chunks, query limits,
  retrieval provenance and answer integration contract;
- image/audio processor chat template, fixed tensor names/dtypes/shapes/bytes,
  IPC identity and malformed-frame rejection;
- browser audio capture state, authenticated loopback STT/TTS APIs, cancellation,
  object-URL cleanup, and fail-closed unavailable-artifact state;
- Lens four-gate authorization, exact endpoint, timeout/size/retry bounds, secret
  redaction, normalization, provenance, attribution, and explicit RAG indexing;
- proposal diff integrity, stale-base rejection, and absence of approve/apply
  authority.

## 동결 소스 GPU·System 4 실측

동결 commit `7039152760920461eddec9b049ebbb2a5ac06f0b`에서 현재 개발 장치
`NVIDIA GeForce RTX 5090 Laptop GPU`로 재실행했다.

| 항목 | 관측값 |
|---|---:|
| E4B-it manifest files | 7 verified |
| CTS depth / arena | 100/100 · 301/301 |
| DEQ residual | 0.0015335083 |
| solver/silent fallback | 0 / 0 |
| peak allocated VRAM | 14.8468876 GiB (gate 16.7 GiB) |
| System 4 stress | 10,000 iterations · convergence 1.0 |
| System 4 p50 / p95 | 2.6971 ms / 3.24152 ms |
| System 4 settled FP/FN | 0.0 / 0.0 |

Raw evidence는 `validation/evidence/gemma4_cts_runtime_v040.json`과
`validation/evidence/system4_stress_v040.json`이다. 이는 목표 RTX 4090 인증,
whole-system O(1), combined multimodal VRAM 또는 제품 전체 latency 증거가 아니다.

Component command:

```powershell
python -m pytest -q `
  tests/test_workspace_capabilities.py `
  tests/test_workspace_capabilities_api.py `
  tests/test_multimodal_processor.py `
  tests/test_multimodal_worker_ipc.py `
  tests/test_audio_worker_ipc.py `
  tests/test_local_voice.py `
  tests/test_local_tts.py `
  tests/test_voice_api.py `
  tests/test_voice_ui_contract.py `
  tests/test_lens_api.py `
  tests/test_proposal_review.py
```

## 최종 package에서 다시 수행할 gate

최종 source 변경이 끝난 동일 commit에서 아래 항목을 다시 실행하고 raw output,
environment, skipped reason, commit/tree digest를 보존해야 한다.

- 전체 `pytest`, Ruff check/format, Node syntax
- 10-turn natural Korean conversation
- 20-turn completion stress
- integrated Gemma/DEQ/CTS depth-100와 System 4는 동결 소스에서 완료됐으며
  package smoke가 동일 commit을 가리키는지 확인
- deterministic local image forward
- loopback UI/API와 Windows launcher smoke
- EXE, wheel, source ZIP, manual/checklist의 final SHA-256

기존 v0.3.x JSON과 binary는 변경된 v0.4.0 source scope를 승인하지 않는다.

## 외부·미완료 gate

- exact RTX 4090 24GB validation과 packet-level egress audit
- independent human-labelled conversation study
- semantic RAG relevance, citation correctness, poisoning and corpus benchmark
- broad image/audio dataset, combined multimodal VRAM, WER/TTS quality study
- user token/terms/attribution review를 갖춘 live Lens validation
- trained DEQ/Fast Weight/System 3 artifacts와 independent held-out evidence
- at least three-seed FP-EWC BWT/FWT 및 production PCAS calibration
- kernel-isolated Self-Harness candidate execution, safe promotion, and
  byte-identical rollback fault injection
- independent license/distribution review, code signing, signed installer/updater

릴리스 빌더는 CycloneDX `SBOM.cdx.json`, 설치 빌드 환경의 선언 라이선스
`THIRD_PARTY_NOTICES.md`, artifact SHA-256과 unsigned 상태를 생성한다. 이는 독립
법률 검토나 코드 서명을 대체하지 않는다.

이 증거가 current scope에서 존재하기 전까지 해당 기능은 `gated`, `advisory`,
`research`, `night_only`, `opt_in_gated`, 또는 `proposal_only` 상태를 유지한다.

## v0.4.0 패키지 검증 기록

commit `5bcfbb457f998beb47b3788204c25de3af546b32`에서 release builder를 실행해
EXE·wheel·source ZIP·manual PDF·CycloneDX SBOM·third-party notices와 SHA-256 manifest를
생성했다. 10개 checksum을 다시 계산해 모두 일치함을 확인했다.

- packaged launcher cold start: 49.924초, launcher exit 0
- 인증된 `/api/state`, workspace capabilities, HTML: HTTP 200
- attachment UI, voice UI, local RAG, image-to-model route: enabled/확인
- 인증된 shutdown 후 backend process 종료
- release 집중 회귀: 99 passed, 1 skipped, 169 subtests passed
- actual local model smoke: 7 files verified, finish reason `stop`, CUDA worker cleanup PASS

원시 구조화 기록은 `validation/evidence/release_bundle_v040.json`이다. 이 기록은 현재 개발
장치의 패키지 실행 증거이며, 독립 clean Windows 재현·코드 서명·RTX 4090 인증으로 확대하지 않는다.
