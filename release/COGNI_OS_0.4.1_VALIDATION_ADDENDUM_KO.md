# Cogni-OS 2.0 Genesis v0.4.1 검증 부록

## 문서 역할

이 파일은 검증 결과 자체가 아니라 v0.4.1의 검증 계약을 설명한다. 실제 PASS 판정은
정확한 commit과 SHA-256에 결합된 외부 raw evidence, 독립 검증 결과와
`BUILD_MANIFEST.txt`를 함께 확인해야 한다. 저장소에 포함된 이 문서만으로 어떤
수용 항목도 `COMPLETED`가 되지 않는다.

## 필수 단계

1. 정확한 source commit과 tree digest를 고정한다.
2. 네트워크와 GPU를 차단한 격리 환경에서 Ruff, format, compile 및 전체 pytest를
   실행한다.
3. 같은 commit archive를 다시 구성해 commit id와 Git tree가 일치하는지 확인한 뒤
   CPU 게이트를 반복한다.
4. root 소유 연구실 스케줄러 예약이 있을 때만 물리 GPU 5 Stage G를 실행한다.
5. source, model, config, device, runtime, completion과 identity pre/post 증거를 하나의
   범위로 검증한다.
6. 독립 verifier가 서명한 detached acceptance bundle로 170개 요구사항을 검증한다.
7. 보호된 offline toolchain에서만 `PublishRelease`를 수행한다.

## 실패 폐쇄 조건

- 예약 증빙이 없으면 GPU를 조회하지 않고 `EXTERNAL_BLOCKER / NOT RUN`으로 끝낸다.
- GPU 5가 아닌 물리 장치, UUID 불일치, 만료되거나 수정 가능한 예약은 거부한다.
- 과거 v0.4.0 증거, 다른 commit, 다른 모델 또는 다른 device 결과는 현재 범위에
  합성하지 않는다.
- 승인된 verifier, signature 또는 toolchain closure가 없으면 검증 게시를 시작하지
  않는다.
- artifact-only 출력은 항상 `release_evidence_status=UNVERIFIED`를 유지한다.

## 증거 보관

raw GPU 로그, source/model snapshot과 detached acceptance 작업 파일은 Git에
커밋하지 않는다. 저장소에는 독립 검토를 거친 content-addressed 요약만 추가할 수
있다. EXE, wheel, ZIP 및 checksum은 생성물이며 source commit과 분리해 배포한다.

## 현재 판정 원칙

CPU 테스트 통과는 소프트웨어 정적·회귀 게이트의 증거일 뿐 GPU 성능, 16.7 GiB
VRAM, CTS depth 100, 자연어 품질 또는 검증 릴리스를 자동 승인하지 않는다. GPU5
예약과 독립 게시 신뢰 경계가 충족되기 전에는 v0.4.1을
`UNVERIFIED artifact-only release candidate`로만 표시한다.
