# Cogni-OS 2.0 Genesis v0.4.0 릴리스 노트

> **과거 범위 문서:** 이 문서는 v0.4.0 당시의 관측과 구현을 기록한다. 현재 v0.4.1
> source는 검증 바이트와 upstream path parser 입력을 결합할 수 없는 ABA 경계 때문에
> production `Gemma4Processor` construction을 fail-closed한다. 아래 image/audio/STT
> 관측은 historical evidence이며 현재 answer authority가 아니다.

## 릴리스 정체성

- 제품 버전: `0.4.0`
- 기본 대화 모델: 로컬 manifest 검증 `gemma4-e4b-it`
- 기본 실행 경계: loopback UI, 단일 GPU worker, `use_cache=False`
- Self-Harness 권한: `proposal_only`
- Windows 실행 파일: `CogniBoard-v0.4.0.exe` (frozen commit에서 최종 생성)

v0.4.0은 기존 Phase 1–11 인지·안전 경계 위에 실제 사용 가능한 로컬 지식
워크스페이스와 multimodal/voice 경로를 연결한 릴리스다. Python 클래스나 UI 표지만
추가한 것을 완료로 간주하지 않고, capability 상태와 실제 증거 범위를 분리한다.

## 구현된 기능

### 1. 영속 첨부와 PDF

- 텍스트, Markdown, CSV, JSON, PDF, PNG, JPEG, WebP 파일을 bounded local
  catalog에 저장한다.
- blob은 content-addressed이며 재시작 때 catalog·파일 크기·SHA-256을 다시
  검사한다.
- 미리보기, 삭제, 선택 재색인, 전체 재색인을 제공한다.
- PDF는 HTTP 프로세스 밖의 로컬 `pypdf` worker에서 추출한다. Windows Job
  Object/POSIX rlimit로 RAM 256MiB·CPU 6초를 제한하고 wall 8초, 최대
  128쪽·256,000자 경계를 적용한다. 추출 실패·timeout·변조는 fail-closed 처리한다.

### 2. AkasicDB 로컬 RAG

- 사용자 저장소 `heosanghun/AkasicDB`의 audited commit
  `a6c8e8ebd487e7cb86079f9804a66aaf0914d1dc`와 세 storage module digest를
  확인한 뒤에만 adapter를 연다.
- Flask demo server, 원격 model download, synthetic dataset, hard-coded answer
  경로는 실행하지 않는다.
- 검색 결과는 attachment id, chunk index, score, source provenance를 가진
  bounded `RetrievalEvidence`로만 대화 prompt에 들어간다.
- 현재 deterministic 256-dimensional lexical vectorizer는 trained semantic
  embedder가 아니다.

### 3. 이미지·오디오 tensor 경로

- manifest-bound `Gemma4Processor`의 multimodal chat template로 image/audio를
  처리한다.
- allowlisted dtype·shape·tensor 이름·전체 byte limit를 갖는 고정 스키마 CPU
  tensor만 worker IPC를 통과한다.
- job, lease epoch, deadline, artifact, session identity를 text 요청과 동일하게
  검증한다.
- 한 번의 요청에 사용자가 명시적으로 선택한 이미지 한 장을 전달한다.
- 개발 장치 actual-model smoke에서 595-byte, 256×256 blue-square PNG에
  `중앙의 큰 도형은 파란색 정사각형입니다.`라고 답했고 terminal stop,
  `generation_mode=cogni_core`, application external call 0을 기록했다.
- 비디오 입력은 구현하지 않았다.

이미지 관측은 고정 도형 한 장만 검증하며 일반 visual reasoning이나
image-plus-depth-100 VRAM 조건을 증명하지 않는다.

### 4. 로컬 음성 입출력

- 사용자 동작 뒤에만 브라우저 마이크 권한을 요청한다.
- 16 kHz mono 16-bit PCM WAV, 최대 30초만 로컬 Gemma STT 경로에 허용한다.
- 마지막 답변 읽어주기는 설치된 Windows System.Speech voice를 사용한다.
- 개발 장치 실제 smoke에서 Microsoft Heami Desktop `ko-KR`가 합성한
  `안녕하세요. 코그니보드 로컬 음성 검증입니다.`를 resident Gemma가 동일하게
  전사했다. normalized similarity 1.0, 5.0187초, STT/TTS application external
  call 0이었다.

이 측정은 단일 합성 문장 경로 확인이며 다화자 WER, 배경소음, multilingual,
실시간 latency, 자연스러움 또는 접근성 품질을 증명하지 않는다.

### 5. Lens 특허·논문 검색

- 일반 웹 검색이나 Lens 웹 페이지 scraping 대신 공식 patent/scholarly API의
  고정 endpoint만 구현했다.
- `online_opt_in`, exact `api.lens.org` allowlist, 사용자 API token, 약관 동의의
  네 gate가 모두 있어야 bounded HTTPS POST를 실행한다.
- provenance와 Lens attribution을 유지하며 사용자가 선택한 결과만 AkasicDB에
  색인한다.
- 배포물은 token이나 약관 동의를 포함하지 않는다. 따라서 v0.4.0은 live Lens
  검색 성공을 주장하지 않는다.

### 6. Self-Harness 검토 화면

- evidence·base digest에 묶인 제안 diff를 읽기 전용으로 표시한다.
- 승인, 실행, source overwrite, promotion, 자동 rollback endpoint는 없다.
- source 변경 감지 시 safe mode로 들어가는 `proposal_only` 경계를 유지한다.

## 완료로 주장하지 않는 항목

- 목표 RTX 4090 24GB에서의 depth-100·16.7 GiB 최종 인증
- trained DEQ/Wproj, Fast Weight, System 3 product artifact와 독립 품질 검증
- trained semantic embedding과 독립 RAG relevance/poisoning benchmark
- video understanding, broad image/audio quality, combined multimodal VRAM gate
- 사용자 token·약관 동의가 필요한 live Lens 검색
- generic web search/browsing
- OS/kernel-isolated candidate 실행과 실제 source self-patching
- 자동 proposal promotion·설치·rollback
- code signing, signed installer/updater, hardware-backed signing identity

## 배포 주의

launcher는 standalone model bundle이 아니라 console-free bootstrapper다. 동일
source tree, Python/CUDA dependencies, 검증된 local model과 manifest가 필요하다.
최종 EXE, wheel, source ZIP, 문서의 byte identity는 frozen v0.4.0 commit에서 생성한
`SHA256SUMS.txt`로 확인한다. 빌더는 `SBOM.cdx.json`,
`THIRD_PARTY_NOTICES.md`, `BUILD_MANIFEST.txt`도 함께 생성하며 코드 서명 인증서가
없다는 사실을 manifest에 표시한다. 이 인벤토리는 독립 라이선스 검토나 법률 자문,
암호학적 서명을 대신하지 않는다. 이전 버전 binary나 JSON을 v0.4.0 증거로
재사용하지 않는다.

상세 gate는 `docs/VALIDATION.md`, 안전 경계는 `docs/SECURITY.md`, 운영 방법은
`docs/COGNIBOARD_USER_MANUAL_PLAYBOOK_KO.md`와 `docs/AGENT_WORKSPACE_KO.md`를
참고한다.
