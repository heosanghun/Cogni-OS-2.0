# CogniBoard v0.4.0 AI 워크스페이스 사용 안내

## 실행

최종 검증 후 생성된 `release\CogniBoard-v0.4.0.exe`를 더블클릭한다. 개발 소스에서
직접 빌드하려면 다음 명령을 사용한다.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_windows_launcher.ps1 `
  -OutputPath release\CogniBoard-v0.4.0.exe
```

launcher는 standalone model bundle이 아니라 console-free bootstrapper다. 동일
source tree, 로컬 Python/CUDA dependency, 검증된 instruction-tuned 모델과 manifest가
필요하다. 제품 대화용 기본 모델 위치는 `C:\Project\cognios\gemma4-e4b-it`이며 다른
위치는 실행 전
`COGNI_OS_MODEL_DIR`로 지정한다. runtime download는 허용하지 않으며 모델 추론의
기본 경로는 외부 API를 사용하지 않는다. 예외는 사용자가 네 가지 gate를 모두
명시적으로 켠 Lens 공식 API 검색뿐이다.
`C:\Project\cognios\gemma4-e4b` base 체크포인트는 명시적인 연구·canary 재현용이며
제품 대화 경로에는 사용할 수 없다.

### AkasicDB 로컬 RAG 준비

RAG는 사용자가 지정한 저장소의 audited revision을 외부 local adapter로 사용한다.

```powershell
git clone https://github.com/heosanghun/AkasicDB.git C:\Project\AkasicDB
git -C C:\Project\AkasicDB checkout a6c8e8ebd487e7cb86079f9804a66aaf0914d1dc
$env:COGNI_OS_AKASICDB_DIR = 'C:\Project\AkasicDB'
```

CogniBoard는 AkasicDB의 Flask 데모 서버, 원격 모델 다운로드, synthetic dataset과
하드코딩 답변을 실행하지 않는다. 고정 commit과 파일 digest가 일치하는
GraphStore/RelationalStore/VectorStore만 bounded adapter를 통해 로드한다. 해당
revision에는 독립 LICENSE 파일이 없으므로 AkasicDB 소스를 Cogni-OS 배포물에
복사하지 않는다.

### 첨부·PDF·RAG 사용

1. 입력창의 `+` 버튼에서 로컬 파일을 선택한다. 허용 확장자는 `.txt`, `.md`,
   `.csv`, `.json`, `.pdf`, `.png`, `.jpg`, `.jpeg`, `.webp`다.
2. 첨부 목록에서 미리보기·삭제·재색인을 명시적으로 실행한다. 카탈로그와 blob은
   로컬 파일시스템에 남으며 재시작 때 크기와 digest를 다시 검사한다.
3. 텍스트·PDF를 AkasicDB에 색인한 뒤 `로컬 RAG`를 켠 요청만 검색 증거를 사용한다.
   PDF는 로컬 `pypdf`로 최대 128쪽·256,000자까지 추출된다.
4. 답변에 사용된 출처는 attachment id, chunk index, score와 함께 표시된다. 현재
   256차원 결정론적 lexical vectorizer는 학습된 semantic embedder가 아니다.

이미지는 첨부 후 한 장을 명시적으로 선택한 다음 한 번의 대화 요청에 전달한다.
선택된 이미지와 RAG는 같은 요청에서 동시에 자동 활성화되지 않는다. 이미지·오디오는
manifest로 묶인 `Gemma4Processor`를 거쳐 고정 스키마 CPU tensor IPC로 resident
worker에 전달된다. 비디오 입력은 구현되어 있지 않다.

개발 장치 actual-model image smoke는 로컬에서 생성한 256×256 PNG의 파란색
정사각형을 `중앙의 큰 도형은 파란색 정사각형입니다.`라고 답했고 terminal stop과
application external call 0을 기록했다. 이는 그 한 장의 고정 도형만 검증하며 일반
시각 추론이나 multimodal VRAM 조건을 증명하지 않는다.

### 로컬 음성

음성 버튼은 사용자가 눌렀을 때만 브라우저 마이크 권한을 요청한다. 캡처된 음성은
16 kHz, mono, 16-bit PCM WAV, 최대 30초로 변환·검사한 뒤 현재 검증 모델에
전달된다. `읽어주기`는 Windows에 설치된 System.Speech 음성을 사용하며 네트워크
TTS를 호출하지 않는다.

개발 장치에서 Microsoft Heami Desktop `ko-KR`로 합성한
`안녕하세요. 코그니보드 로컬 음성 검증입니다.`를 resident Gemma가 동일하게
전사해 normalized similarity 1.0, application external call 0을 기록했다. 이는
단일 합성 문장 smoke일 뿐 다화자 WER, 소음 강건성, 실시간 지연, 접근성 품질을
증명하지 않는다.

### Lens 특허·논문 검색

Lens 검색은 일반 웹 검색이나 웹 페이지 scraping이 아니다. 아래 네 조건을 모두
명시적으로 충족한 경우에만 공식 `api.lens.org` API에 bounded HTTPS POST를 보낸다.

```powershell
$env:COGNI_OS_NETWORK_MODE = 'online_opt_in'
$env:COGNI_OS_WEB_ALLOWLIST = 'api.lens.org'
$env:COGNI_OS_LENS_API_TOKEN = '<사용자에게 발급된 토큰>'
$env:COGNI_OS_LENS_TERMS_ACCEPTED = '1'
```

토큰이나 약관 동의를 배포물에 포함하지 않는다. 결과는 Lens provenance를 유지하며
사용자가 선택한 경우에만 AkasicDB에 색인된다. v0.4.0 배포 검증은 실제 사용자
토큰으로 live Lens 검색을 수행했다고 주장하지 않는다. 공개 배포 전 Lens 이용약관,
attribution, 승인된 logo 자산 사용 조건을 운영자가 별도로 확인해야 한다.

## 대화

1. manifest와 Runtime Fact-book을 검증한다.
2. 하나의 CUDA worker와 lease를 사용한다.
3. Gemma feature를 bounded CTS/DEQ canary에 전달한다.
4. 안전한 terminal latent가 non-zero bounded logits bias로 답변에 기여한다.
5. System 3/4는 advisory telemetry로만 실행된다. 학습된 Fast Weight artifact가
   없으면 System 1.5는 gated 상태다.
6. 일반 대화는 bounded sampling, 정확한 문장 수 요청은 grounded strict decode를
   사용하며, `use_cache=False` 뒤 반복·역할 토큰·거짓 정체성·미완성 문장을 검사한다.
7. 완전한 답변만 대화 기록에 commit한다. 실패 시 복구 generation은 최대 한 번이다.

답변이 중간에서 잘리거나 반복 guard가 동작한 경우 UI는 정상 완료로 표시하지 않고
finish/quality reason을 공개한다. Agent의 모델 크기와 기능 설명은 언어 모델의 기억이
아니라 현재 artifact의 Fact-book을 기준으로 한다.

## 작업

결정론적으로 해석 가능한 slash command와 안전 문장만 immutable TypedTaskPlan으로
변환된다. 대표 명령은 `/help`, `/list`, `/read`, `/search`, `/status`, 고정 `/test`,
output-only `/save`다.

- T0: bounded read/list/search/status
- T1: 고정 pytest와 `outputs/agent-workspace` 산출물
- T2: source를 바꾸지 않는 inert Self-Harness proposal staging
- T3: network, arbitrary shell, evaluator/security/updater mutation 영구 거부

절대/UNC/device 경로, `..`, ADS, junction/symlink escape, 임의 executable은
거부한다. 모델의 자유 형식 plan은 실행 권한이 아니다.

## Self-Harness

실패와 성공을 bounded local evidence로 저장하고, 정확한 causal signature별로 최소
세 개의 서로 다른 후보가 모일 때만 rich proposal을 만든다. 모든 후보는 primary
evidence, expected behavior, risk, reproduction test와 rollback trigger를 가진다.
거부된 후보는 negative archive에 남는다.

v0.4.0 지원 profile은 `proposal_only`다. source는 실행·덮어쓰기·승격되지 않는다.
UI의 제안 diff는 읽기 전용이며 승인·적용·promotion·rollback 버튼을 제공하지 않는다.
야간 주기 전후 source digest가 달라지면 safe mode로 진입한다. 별도 kernel, network/
host-filesystem isolation, immutable image와 command digest가 독립 검증되는 Phase 12
이전에는 자동 승격을 기능으로 표시하지 않는다.

## 재현

```powershell
python scripts\validate_agent_completion.py `
  --model C:\Project\cognios\gemma4-e4b-it `
  --manifest config\gemma4-e4b-it.manifest.toml

python scripts\validate_agent_casual_korean.py `
  --model C:\Project\cognios\gemma4-e4b-it `
  --manifest config\gemma4-e4b-it.manifest.toml `
  --timeout 120 `
  --output C:\Project\cognios-evidence\casual-korean-v0.4.0.json

python -m scripts.validate_gemma4_local_voice `
  --model C:\Project\cognios\gemma4-e4b-it `
  --manifest config\gemma4-e4b-it.manifest.toml `
  --output C:\Project\cognios-evidence\local-voice-v0.4.0.json

python -m scripts.validate_gemma4_local_image `
  --model C:\Project\cognios\gemma4-e4b-it `
  --manifest config\gemma4-e4b-it.manifest.toml `
  --output C:\Project\cognios-evidence\local-image-v0.4.0.json

python scripts\validate_agent_runtime.py `
  --model C:\Project\cognios\gemma4-e4b-it `
  --manifest config\gemma4-e4b-it.manifest.toml `
  --prompt "사용자에게 자연스러운 한국어 한 문장으로 인사하세요." `
  --max-new-tokens 64
```

`validate_agent_runtime.py`는 채팅 직렬화·공개 응답 채널·종료 토큰·한 턴 품질
계약과 작업자 정리를 확인하는 저수준 GPU smoke다. 전체 자연 대화 품질의 근거는
별도의 10턴 casual gate와 20턴 completion stress 결과를 사용한다.

첫 명령은 네 번의 형식 중심 실제 오프라인 대화를 검사한다. 두 번째 명령은 보고된
협업 대화 원문, 오타, 바꿔 말하기, 후속 질문과 문맥 전환을 포함한 10턴에서
fallback 0회와 자연스러운 완결성을 검사한다. 어느 명령도 RTX 4090 인증을 대신하지
않는다. 최신 scoped GPU 관측과 정확한 제한은
[GEMMA4_VALIDATION.md](GEMMA4_VALIDATION.md)를 참고한다.
