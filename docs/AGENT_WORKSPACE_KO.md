# CogniBoard AI 워크스페이스 사용 안내

## 실행

저장소 루트의 `CogniBoard.exe`를 더블클릭합니다. 콘솔 진단이 필요하면
`Run-CogniOS-Demo.cmd`를 실행합니다. 기본 모델 위치는
`C:\Project\cognios\gemma4-e4b`이며 다른 위치를 쓰려면 실행 전에
`COGNI_OS_MODEL_DIR` 환경 변수를 설정합니다.

서버는 브라우저를 열기 전에 manifest에 선언된 로컬 모델 6개 파일의
SHA-256을 검사합니다. 첫 대화는 단일 CUDA worker에서 모델과 Cogni-Core를
적재하므로 장치와 디스크 상태에 따라 약 1분 이상 걸릴 수 있습니다. 이후
worker가 살아 있는 동안에는 재적재하지 않고 토큰을 순차 표시합니다.

## 대화 모드

대화는 다음 실제 경로를 거칩니다.

1. bounded 멀티턴 기록과 로컬 chat template
2. BIO-HAMA 인지 라우팅
3. Gemma feature backbone과 DEQ/CTS Depth 100
4. System 4 텐서 swarm과 System 3 bounded expert 관측
5. `use_cache=False`, deterministic Gemma 답변 생성
6. CPU `int64` 토큰 텐서 스트리밍과 대화 commit

System 4와 System 3는 현재 advisory-only입니다. 검증되지 않은 보조 모듈이
Gemma의 답변 토큰을 바꾸지 못합니다. Fast Weight는 외부 품질·OOD·스펙트럼
게이트를 통과한 세션 overlay가 없으므로 기본 비활성이고, FP-EWC는 주간 대화
경로에서 제외됩니다.

## 작업 모드

작업 모드는 임의 셸 문장을 실행하지 않습니다. 아래 명시 명령만 허용합니다.

- `/help`
- `/list [상대경로]`
- `/read <상대파일>`
- `/search <검색어> [--in <상대경로>]`
- `/status`
- `/test [tests/파일.py]`
- `/save <파일명>` 뒤에 저장할 내용

`/save` 결과는 `outputs/agent-workspace`에만 기록됩니다. 절대경로, `..`,
symlink/junction, 바이너리, 과대 파일, 소스 확장자 저장은 거부됩니다. 소스
변경은 작업 모드가 아니라 Self-Harness 경계를 통해서만 가능합니다.

## Self-Harness

실행 실패와 timeout은 bounded SQLite에 수집됩니다. `안전 진화 주기 실행`은
활성 대화와 검증이 모두 끝난 뒤에만 시작되며, 로컬 Gemma가 운영자 지정
실패 signature와 소스 allowlist를 기반으로 inert patch proposal을 만듭니다.

현재 기본 제품 구성은 `proposal_only`입니다. 이 PC에서 별도 커널, 네트워크
차단, host filesystem 차단, ephemeral workspace를 증명하는 실행기가 확인되지
않았기 때문에 UI나 모델이 임의로 소스를 덮어쓰지 않습니다. 실제 자동 승격은
운영자가 독립 감사한 runner id/evidence digest와 정확한 회귀·health 명령 digest를
신뢰 목록에 넣은 경우에만 켜집니다. 그때도 staging 회귀, backup journal,
atomic replace, 별도 health snapshot, SHA-256 rollback을 모두 통과해야 합니다.

## 재현 검증

```powershell
python scripts\validate_agent_runtime.py `
  --model C:\Project\cognios\gemma4-e4b `
  --manifest config\gemma4-e4b.manifest.toml `
  --prompt "Cogni-OS 안전 경계를 설명해 주세요." `
  --max-new-tokens 64
```

2026-07-11 내부 RTX 5090 Laptop GPU 재현에서는 전체 Cogni-Core+Gemma 경로가
통과했고, WDDM 총 GPU 메모리의 실행 전 대비 관측 증가는 15.397 GiB였습니다.
worker 종료 후 실행 전 기준값으로 복귀했습니다. 이는 이 장비의 내부 실측이며
목표 RTX 4090 결과나 AGI 성능 주장이 아닙니다.
