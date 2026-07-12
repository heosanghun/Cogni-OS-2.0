# Cogni-OS 2.0 Genesis v0.3.2 검증 부록

## 승인 원칙

테스트의 종료 코드만 보지 않고 실제 답변 원문을 다시 감사한다. 다음 항목 중 하나라도
실패하면 릴리스 상태는 `FAILED`다.

1. 사용자 요청당 assistant 답변이 정확히 하나다.
2. 답변에 역할·제어 토큰, 동일 문장 반복, 미완성 꼬리가 없다.
3. 지정된 문장 수에서 번호 표식은 문장으로 계산하지 않는다.
4. 질문 주제와 최소 관련성 계약을 지키며 7B·외부 계정·성공 결과를 만들지 않는다.
5. `quality_fallback`을 정상 완료로 표시하지 않는다.
6. 요청과 worker의 전체 deadline이 제한되고 취소 후 채널이 정상 drain된다.
7. 종료 후 resident worker와 GPU lease가 정리된다.

## 필수 증거

- 전체 pytest/ruff/format/Node 문법 결과
- `scripts/validate_agent_casual_korean.py`의 10/10 실GPU JSON
- `scripts/validate_agent_completion.py --turns 20`의 실GPU JSON
- 현재 프로세스의 UI/API smoke와 버전·Fact-book 일치
- 동결 commit에서 생성한 EXE, wheel, source ZIP, manual PDF와 `SHA256SUMS.txt`

과거 v0.3.0/v0.3.1 JSON은 회귀 비교 자료일 뿐 v0.3.2 승격 권한이 없다. 최종 JSON과
배포 byte는 동일한 v0.3.2 commit에서 생성해야 한다.
