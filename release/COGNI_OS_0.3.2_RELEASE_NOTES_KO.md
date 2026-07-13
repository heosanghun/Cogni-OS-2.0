# Cogni-OS 2.0 Genesis v0.3.2 릴리스 노트

## 핵심 수정

v0.3.2는 로컬 Gemma 대화가 중간에 끊기거나 같은 문장을 반복하고, 짧은 인사·협업
제안에서 엉뚱한 학습 문구로 이탈하던 문제를 제품 경계에서 수정한다.

- Gemma 4 공식 `turn/model` 프롬프트와 thought-channel 공개 답변 파서를 고정했다.
- 일반 대화는 모델 카드의 bounded sampling을 사용하고, 정확한 문장 수 요청은
  요청 주제 prefill과 strict decode를 사용한다.
- 반복·역할 표기·제어 토큰·미완성 문장·주제 침입·거짓 외부 서비스 문구를 게시 전
  차단한다.
- 문장 번호를 실제 문장으로 잘못 세지 않으며, 완전한 번호 절만 보수적으로 문장으로
  정규화한다.
- 인사·협업 제안·능력 안내·직전 협업에 이어지는 첫 질문은 제한된
  `conversation_fastpath`로 한 번만 응답한다. 일반 지식·코드·형식 요청은 반드시
  Fact-book 또는 Cogni-Core 경로로 내려간다.
- Runtime Fact-book 원문은 일반 모델 프롬프트에서 제외해 제품 메타데이터 복사를
  방지한다.
- worker 전체 deadline, cancel drain, poisoned-worker retire와 GPU lease 해제를
  fail-closed로 처리한다.
- UI는 실제 Gemma, Fact-book, Fast Path, 품질 실패를 서로 다른 배지로 표시한다.

## 검증 명령

```powershell
python -m ruff check .
python -m ruff format --check cogni_agent cogni_core cogni_demo cogni_flow cogni_os scripts tests
python -m pytest -q

python scripts\validate_agent_casual_korean.py `
  --model C:\Project\cognios\gemma4-e4b-it `
  --manifest config\gemma4-e4b-it.manifest.toml `
  --timeout 120 `
  --output C:\Project\cognios-evidence\casual-korean-v0.3.2.json

python scripts\validate_agent_completion.py `
  --model C:\Project\cognios\gemma4-e4b-it `
  --manifest config\gemma4-e4b-it.manifest.toml `
  --turns 20
```

10턴 자연 대화 게이트, 20턴 실모델 completion stress, 전체 회귀, UI HTTP smoke,
릴리스 checksum이 모두 통과한 동일 commit만 v0.3.2 배포 후보로 인정한다.
공개 대화에는 공식 instruction-tuned E4B-it와 코드에 고정된 정확한 매니페스트만
허용한다. pretrained base E4B는 연구·canary 전용이며 제품 대화 경로에서 거부한다.

## 경계

`conversation_fastpath`는 학습된 Fast Weight가 아니며 System 1.5의 활성 증거가
아니다. System 1.5는 gated, System 3/4는 advisory, Self-Harness는 proposal-only다.
현재 실측 RTX 5090 Laptop 결과는 목표 RTX 4090 인증으로 승격되지 않는다.
