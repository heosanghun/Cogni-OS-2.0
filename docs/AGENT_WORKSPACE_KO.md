# CogniBoard v0.3.2 AI 워크스페이스 사용 안내

## 실행

최종 검증 후 생성된 `release\CogniBoard-v0.3.2.exe`를 더블클릭한다. 개발 소스에서
직접 빌드하려면 다음 명령을 사용한다.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_windows_launcher.ps1 `
  -OutputPath release\CogniBoard-v0.3.2.exe
```

launcher는 standalone model bundle이 아니라 console-free bootstrapper다. 동일
source tree, 로컬 Python/CUDA dependency, 검증된 instruction-tuned 모델과 manifest가
필요하다. 제품 대화용 기본 모델 위치는 `C:\Project\cognios\gemma4-e4b-it`이며 다른
위치는 실행 전
`COGNI_OS_MODEL_DIR`로 지정한다. runtime download와 외부 API는 허용하지 않는다.
`C:\Project\cognios\gemma4-e4b` base 체크포인트는 명시적인 연구·canary 재현용이며
제품 대화 경로에는 사용할 수 없다.

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

v0.3.2 지원 profile은 `proposal_only`다. source는 실행·덮어쓰기·승격되지 않는다.
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
  --output C:\Project\cognios-evidence\casual-korean-v0.3.2.json

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
