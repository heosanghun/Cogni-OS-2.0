# Cogni-OS 2.0 Genesis v0.2.1 릴리스 노트

## 해결한 현상

CogniBoard의 로컬 AI 답변이 문장 중간에서 끝나거나, 답변 안에 `USER:`와
`ASSISTANT:`가 나타나며 이전 대화를 계속 생성하는 문제를 해결했습니다.

확정된 원인은 다음과 같습니다.

1. 로컬 Gemma 4 미러에 `chat_template.jinja`가 없어 잘못된 평문 역할 형식으로
   후퇴했습니다.
2. AgentManager의 192토큰 한도를 EOS와 구분하지 않고 성공으로 기록했습니다.
3. Gemma 4의 `<turn|>` 종료 토큰을 worker에 전달하지 않았습니다.
4. 180초를 전체 요청 절대시간으로 사용하여 정상 스트리밍도 중단될 수 있었습니다.
5. 일부 예약 토큰이 special token으로 등록되지 않아 완결된 본문 뒤에서 반복됐습니다.
6. 전체 시퀀스 hidden state를 CTS 301개 노드마다 사전 할당하여 후속 대화의 VRAM
   입장량이 문맥 길이에 비례했습니다.

## 구현 변경

- 공식 텍스트 전용 Gemma 4 턴 형식을 로컬 코드에 고정하고 BOS를 정확히 한 번만
  삽입합니다.
- protocol v2가 `stop`, `length`, `cancelled`, `error` 종료 사유를 텐서로 전달합니다.
- EOS·EOT·tool handoff와 로컬 reserved token을 공개 응답 전에 차단합니다.
- 512토큰 세그먼트가 `length`로 끝날 때만 같은 model turn에서 자동으로 이어 쓰며,
  최대 2회·총 1,536토큰으로 제한합니다.
- 타임아웃은 토큰 수신마다 갱신되는 worker-idle 기준으로 변경했습니다.
- 8토큰 또는 50ms 단위로 화면을 갱신하고, 토큰이 없는 최종 프레임도 반드시
  flush합니다.
- 출력 한도를 8,192자로 분리하고 `finish_reason`, `continuations`, `truncated`,
  `generated_tokens`를 UI에 표시합니다.
- Cogni-Core의 advisory 입력을 attention-weighted 고정 크기 Gemma embedding으로
  풀링하여 CTS arena가 문맥 길이에 따라 커지지 않도록 했습니다.

## 검증 결과

- 전체 자동 회귀: **264 passed, 1 skipped, 133 subtests passed**
- Ruff: **통과**
- JavaScript 구문 검사: **통과**
- 실제 로컬 모델: `C:\Project\cognios\gemma4-e4b`
- 실제 장치: NVIDIA GeForce RTX 5090 Laptop GPU
- manifest 검증 파일: 6개
- 실제 3턴 연속 대화:
  - 짧은 응답: 8토큰, `stop`, truncation 없음
  - 기능 설명: 143토큰, `stop`, truncation 없음
  - 후속 요약: 129토큰, `stop`, truncation 없음
- 3턴 모두 역할 표기 누출 0건, 공개 제어 토큰 0건
- 종료 후 resident model worker 정상 해제

## 역사적 재현 상태

당시의 직접 GPU 명령은 현재 서버 안전 계약을 우회하므로 제거했다. 이 기록은
historical observation이며 현재 실행 지침이 아니다. GPU-bearing 검증은 깨끗한
동일 커밋 CPU Exit Gate 이후 `docs/VALIDATION.md`의 GPU5 guard 경로로만 허용한다.

본 결과는 현재 연결된 RTX 5090 Laptop GPU의 내부 실측이며 RTX 4090 인증 결과로
표시하지 않습니다.
