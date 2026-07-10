# Cogni-OS 2.0 (Genesis) 최종 구현·검증 보고서

기준일: 2026-07-10  
소스 버전: `0.1.0`  
구현 규모: Python 60개 파일, 12,259줄, 테스트 파일 27개

## 1. 최종 판정

요청하신 문헌 기반 모듈을 하나의 **실행 가능한 오프라인 연구 runtime**으로
구현했습니다. 세 선행 수학 gate, CTS, Fast Weight, FP-EWC, System 3/4,
BIO-HAMA, AFlow, Self-Harness, 실패 daemon, 야간 scheduler, 로컬 모델 proposer,
tensor-only 프로세스 경계까지 코드와 테스트로 연결했습니다.

최종 자동 검증 결과는 다음과 같습니다.

- Ruff 정적 검사: 통과
- Ruff format 검사: 60개 파일 통과
- 회귀 테스트: **154개 통과**
- Python warning을 error로 승격한 실행: 통과
- wheel build: 통과
- 소스 트리 밖 임시 환경에서 설치형 `doctor`: 통과
- 설치형 internal self-check: 통과
- 실제 로컬 Gemma 4 일반 forward: 통과
- 실제 Gemma 4 마지막 레이어 DEQ 수렴 smoke: 통과(비인증 실험 모드)
- 실제 Gemma + 전체 보조 runtime + CTS depth 100: 통과

다만 이것을 “AGI 완성”, “전체 Gemma의 수학적 수축성 증명”, 또는 “무제한
자율 진화의 실증”이라고 부르지는 않습니다. 아래 미검증 항목을 명시적으로
fail-closed 처리했습니다.

## 2. 최초 자료 조사 결과

### PDF

`C:\Project\Self-Harness\doc`의 PDF 10개, 총 270쪽을 모두 추출·분석했습니다.
특허 2개와 CTS, System 1.5/2.5/3/4, AFlow, ADAS, Self-Harness를 하나의 구현
추적표로 연결했습니다. 세부 분석은 `PDF_CORPUS_ANALYSIS_KO.md`에 있습니다.

### GitHub

`heosanghun` 계정의 공개 저장소를 최종 재조회한 결과 55개였습니다. 55개를
전부 목록화·분류하고, Cogni-OS 직접 관련 11개를 shallow clone해 소스 수준으로
검토했습니다: CTS 2개, System1.5 2개, System2.5, System3/3.5, System4/5,
BIO-HAMA_MAIN, AkasicDB. 전체 목록과 반영 내용은
`GITHUB_REPOSITORY_AUDIT_KO.md`에 있습니다.

## 3. 구현된 시스템

| 영역 | 구현 내용 | 핵심 안전 경계 |
|---|---|---|
| DEQ | limited-history multisecant/Broyden, IFT backward, fallback | 미수렴·비유한·비수축 기본 거부 |
| Gemma | local-path 전용 loader, hash manifest, Gemma 4 layout adapter | Hub/URL/remote code 차단, 16.7 GiB 사전·사후 gate |
| CTS | fixed 301-node PUCT arena, shared latent ancestor bank, frontier rollout | `max_depth`가 tensor allocation을 키우지 않음, no-KV 계약 |
| Fast Weight | bounded bottleneck programmer, session overlay, CPU LRU | base 불변, 외부 품질, delta 및 합성 operator norm, OOD fallback |
| FP-EWC | matrix-free fixed-point Fisher, bounded domain merge | optimizer 전후 C-FIRE, restore 후 재투영 |
| System 3 | preallocated sparse expert pool, top-k, novelty recruitment | 무제한 spawn 금지, recycle/merge, 3D spectral projection |
| System 4 | tensor swarm, PCAS change score, precompiled topology | 가중치 update 없는 topology 전환, recurrent/coupling C-FIRE |
| BIO-HAMA | 5요소 인지 tensor, 전략·전술·반응 mask, alpha/gamma | hot path tensor fields, 고정 module budget |
| AFlow | immutable workflow DAG, soft-mixed selection, bounded search/archive | 생성 payload inert, 실행 권한 없음 |
| Self-Harness | failure clustering, AST/digest policy, staging, regression, atomic replace | kernel-isolated runner 없으면 실행 전 거부 |
| Phase 4 운영 | async failure daemon, monotonic idle scheduler, local Gemma proposer | bounded queue, 중복 야간 cycle 차단, 모델은 target path 선택 불가 |
| MSA 경계 | Windows-spawn `TensorService`, numeric PAUSE/RESUME/INFER/STOP | 요청·응답이 정확히 CPU Tensor 4개, bounded queue, socket 없음 |
| 배포 | wheel, package default config, CLI doctor/validate | 현재 작업 폴더에 의존하지 않음 |

## 4. 세 선행 검증과 전체 테스트

1. 암시적 DEQ forward와 IFT gradient가 250-step explicit unroll과 허용 오차
   안에서 일치합니다.
2. CUDA CTS active allocation이 depth 8과 64 사이에서 설정한 8 MiB 허용량을
   넘지 않습니다.
3. spectral margin을 넘는 연산자는 기본 hard-stop하며, 연구용 fallback은
   명시적으로 opt-in해야 합니다.

추가로 다음을 검증했습니다.

- 요청 depth 2와 10,000,000에서 고정 node capacity의 사전 할당 bytes 동일
- BF16 ancestor blend dtype 보존
- BF16 C-FIRE의 FP32 exact-norm 계산과 원본 dtype 보존
- Fast Weight base+overlay 보수적 합성 norm `< 0.95`
- expert/swarm 3D batch matrix 개별 투영
- checkpoint key/shape/dtype/finite 검사와 복원 후 C-FIRE
- evolution 작업 중 inference 재개 race 차단
- process-only sandbox 실행 전 거부
- daemon overflow/writer failure fail-closed
- tensor service pause/resume, timeout, capacity, restart, numeric error isolation

최종 명령:

```powershell
python -m ruff check .
python -m ruff format --check cogni_core cogni_flow cogni_os tests scripts
python -W error -m unittest discover -s tests
```

결과: `Ran 154 tests ... OK`.

## 5. 실제 Gemma 4 측정

모델 경로: `C:\Project\cognios\gemma4-e4b`  
모델: `Gemma4ForConditionalGeneration`, BF16, decoder 42층, hidden 2560  
GPU: NVIDIA GeForce RTX 5090 Laptop GPU, 물리 VRAM 23.89 GiB  
정책 상한: 16.7 GiB

### Artifact

- manifest 검증 파일: 6개
- `model.safetensors`: 15,992,595,884 bytes
- SHA-256: `43fb96cec3045b72852c787540300dc5b258634b7a025f7c80355ac0788b9651`

### 일반 forward

- load: 25.981 s
- forward: 0.823 s
- logits: `(1, 8, 262144)`, finite
- peak allocated: **14.8955 GiB**

### Gemma 마지막 레이어 DEQ smoke

- `contractive_delta_scale=0.05`
- 반복: 4회
- normalized residual: `0.0018844604`
- fallback: 미사용
- forward: 0.975 s
- logits: `(1, 10, 262144)`, finite
- peak allocated: **14.8955 GiB**
- 판정: 수렴 smoke 통과, 그러나 decoder delta의 전역 Lipschitz 인증값은 없음

### 최종 통합 runtime

최종 factory에는 Gemma, Fast Weight 병목 adapter/programmer, System 3/4,
BIO-HAMA가 함께 적재됐습니다. 그 상태에서 전역 수축계수 0.4인 실제 limited-
Broyden latent transition으로 CTS를 실행했습니다.

- load: 22.500 s
- integrated inference: **4.636 s**
- requested/reached depth: **100/100**
- nodes: **301/301**
- fixed search allocation: **14,994,009 bytes**
- 마지막 transition residual: `0.0039062500`
- fallback: 미사용
- backbone/search latent: finite
- integrated inference peak: **14.8560 GiB**

프로세스 전체의 가장 높은 관측값은 모델 load gate의 14.8955 GiB로, 16.7 GiB
상한보다 약 1.80 GiB 낮습니다. 이 수치는 prompt 길이와 현재 software stack에
대한 측정이며 RTX 4090 결과로 간주할 수는 없습니다.

## 6. 절대 규칙별 판정

### Rule 1 — VRAM 16.7 GiB

설정에서 16.7보다 높은 값은 거부됩니다. 모델 weight byte와 현재 allocation을
적재 전에 admission하고, CUDA OOM을 budget error로 변환하며, 모든 guarded
구간 뒤 peak를 확인합니다. CTS는 depth가 아니라 고정 node capacity로 tensor를
할당합니다. 실제 최종 통합 측정은 통과했습니다.

정확한 표현은 “고정 capacity에서 depth-independent working set”입니다. 모델
가중치와 모든 외부 데이터까지 포함한 무조건적 O(1) total memory 주장은 하지
않습니다.

### Rule 2 — Tensor 중심 통신

CTS/System3/System4/BIO-HAMA hot path와 별도 프로세스 data plane은 Tensor만
사용합니다. `TensorService`는 4-Tensor numeric protocol을 강제합니다. 파일 경로,
session 이름, 감사 문자열은 의도적으로 Cogni-Flow/control boundary에만 존재합니다.

### Rule 3 — 주/야간 배타

active inference와 active evolution counter, checkpointing, validating,
promoting, rollback 상태를 강제합니다. proposer/AFlow/patch promotion 전체 수명
동안 inference 복귀를 차단하고, scheduler는 중복 night cycle을 거부합니다.

### Rule 4 — C-FIRE

2D 및 3D+ 마지막 두 차원 행렬을 개별 투영합니다. expert/swarm/adapter 연산 직전,
optimizer 전후, checkpoint restore 후에 적용합니다. BF16 norm은 FP32 exact SVD로
측정하고 가중치 dtype은 보존합니다.

Gemma 전체 decoder의 전역 `L <= 0.95`는 아직 인증되지 않았으므로 production
Gemma-DEQ 경로는 인증값 없이는 생성 단계에서 fail-closed됩니다.

### Rule 5 — Air-gap

runtime은 URL/Hub ID, remote code, force download, network client import를
거부합니다. 모델과 tokenizer는 로컬 path와 manifest로만 로드됩니다. 로컬 proposer도
이미 로드된 객체만 받으며 자체 model loader나 network client가 없습니다.

## 7. 의도적으로 차단된 항목

다음은 코드 누락을 성공처럼 숨기지 않고 명시적 gate로 남겼습니다.

- **RTX 4090 실측:** 현재 장치는 RTX 5090 Laptop입니다. 4090에서 같은 세 명령을
  재실행해야 합니다.
- **Gemma 전역 수축 인증:** 한 입력에서의 수렴은 전역 Lipschitz 증명이 아닙니다.
  인증 artifact가 없으면 production Gemma-DEQ 주입이 거부됩니다.
- **Fast Weight 의미 품질:** runtime 연결과 안전 gate는 완료됐지만, 실제 Gemma
  task에서 정답 품질을 보존하는 overlay 학습/held-out benchmark는 아직 없습니다.
- **FP-EWC/System3 장기 효능:** 수학·component gate는 통과했지만 수십 도메인
  장기 망각률과 expert recycle 품질 benchmark는 별도 실험입니다.
- **자율 패치 production 실행:** 실제 kernel-isolated Windows Sandbox/VM/container
  runner가 주입되지 않았습니다. 따라서 현재 기본 patcher는 후보 실행 전에
  거부합니다. 안전상 올바른 상태입니다.
- **실제 Gemma MSA worker:** tensor service는 CPU spawn protocol로 검증했습니다.
  실제 Gemma는 중복 GPU owner를 만들지 않기 위해 in-process 통합 gate로 측정했습니다.
- **AGI:** 본 결과는 강한 연구 runtime이지 AGI의 과학적 증명이 아닙니다.

## 8. 재현 명령

```powershell
python scripts/validate_gemma4.py `
  --model C:\Project\cognios\gemma4-e4b `
  --manifest config\gemma4-e4b.manifest.toml

python scripts/validate_gemma4_deq.py `
  --model C:\Project\cognios\gemma4-e4b `
  --manifest config\gemma4-e4b.manifest.toml `
  --allow-uncertified-experimental

python scripts/validate_gemma4_runtime.py `
  --model C:\Project\cognios\gemma4-e4b `
  --manifest config\gemma4-e4b.manifest.toml
```

설치형 확인:

```powershell
pip install --no-deps cogni_os-0.1.0-py3-none-any.whl
cogni-os doctor
cogni-os validate
```

오프라인 환경에는 검증에 사용한 `torch 2.11.0+cu128`, `transformers 5.12.1`,
`accelerate 1.14.0`, `safetensors 0.8.0` 또는 호환 버전이 미리 설치돼 있어야
합니다.

## 9. 배포 Artifact

- 소스 archive: `Cogni-OS-2-Genesis-source.zip`
  - SHA-256: `eee7179c151f42375237a25e0fe894fb16c60db2c4e09347e9cd0b03c4bdae55`
- wheel: `cogni_os-0.1.0-py3-none-any.whl`
  - SHA-256: `185fb32b6825e49f8bf08fb11a53f47c601ea2efa227f1285d450335caddb607`
- PDF 분석: `PDF_CORPUS_ANALYSIS_KO.md`
- GitHub 감사: `GITHUB_REPOSITORY_AUDIT_KO.md`
- checksum 목록: `SHA256SUMS.txt`

모델 weight는 크기와 라이선스·배포 경계를 존중해 archive에 포함하지 않았습니다.
