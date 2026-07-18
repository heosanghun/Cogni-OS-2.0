# Cogni-OS 2.0 Genesis v0.4.1 릴리스 노트

## 릴리스 정체성

- 제품 버전: `0.4.1`
- 기본 백본: 로컬 manifest 검증 `gemma4-e4b-it`
- 목표 장치: RTX 4090 24GB
- 서버 연구실 장치 경계: 물리 GPU 5만 허용, GPU 0-4 및 GPU 6-7 금지
- Self-Harness 권한: `proposal_only`
- 외부 네트워크: 기본 차단, 사용자가 명시적으로 활성화한 제한 기능만 별도 허용

v0.4.1은 기능 이름이 아니라 검증 경계를 제품의 일부로 만드는 릴리스 후보이다.
소스, 모델, 설정, 장치, 실행 결과와 대화 완결성을 하나의 증거 범위에 묶되, 현재
확인되지 않은 기능을 완료로 승격하지 않는다.

## 주요 변경

### 1. 대화 완결성과 자연스러움

- 응답 중단, 역할 토큰 노출, 동일 문단 반복과 미완결 한국어 문장을 독립적으로
  판정한다.
- 일반 대화와 엄격한 형식 요청을 분리하고, 이어쓰기 요청에는 한 번의 제한된 추가
  예산만 부여한다.
- 품질 검사를 통과하지 못한 후보는 성공 응답으로 게시하지 않는다.

### 2. 정확한 런타임 사실

- 제품 버전, 모델 형식, effective 및 stored parameter 수, capability 상태를 Runtime
  Fact-book에서 설명한다.
- 과거 장치 결과와 현재 프로세스 실측을 구분하며, 현재 범위에서 검증하지 않은 GPU,
  VRAM, CTS 깊이 또는 품질 수치를 재사용하지 않는다.

### 3. GPU5 전용 서버 경계

- 연구실 스케줄러가 발급한 root 소유 예약 증빙이 없으면 Docker 실행과 첫 GPU 조회
  전에 실패한다.
- 물리 GPU 5와 고정 UUID만 허용하며 다른 GPU를 탐색하거나 대체 장치로 사용하지
  않는다.
- 주간 추론과 야간 진화는 같은 GPU에서 동시에 실행하지 않는다.

### 4. 수용 기준과 릴리스 신뢰 경계

- 170개 요구사항을 `COMPLETED`, `IMPLEMENTED_UNVERIFIED`, `PARTIAL`,
  `NOT_IMPLEMENTED`, `EXTERNAL_BLOCKER`로 보수적으로 관리한다.
- raw CPU/GPU 증거는 저장소 밖 owner-only 위치에 보관하고, 독립 검증된 detached
  bundle만 완료 승격에 사용할 수 있다.
- 일반 artifact-only 빌드는 `UNVERIFIED`로 표시한다. 보호된 toolchain, 승인된
  verifier, 서명된 증거가 없으면 `VERIFIED` 게시를 차단한다.

## 현재 외부 차단 조건

- `/run/cognios-lab-scheduler/gpu5-reservation.json`이 없으면 GPU5 Stage G는
  `EXTERNAL_BLOCKER / NOT RUN`이다.
- release verifier와 toolchain 정책이 `unconfigured`이면 검증 릴리스 게시가
  불가능하다.
- 코드 서명 인증서와 독립 서명 주체가 없으므로 Windows EXE는 unsigned 상태이다.

이 조건들은 구현 오류를 숨기기 위한 면책이 아니다. 외부 권한 또는 독립 신뢰 주체가
필요한 경계를 소스 코드가 임의로 우회하지 못하게 하는 안전 조건이다.

## 배포 판정

`CogniBoard-v0.4.1.exe`, wheel, source ZIP, SBOM, notices와 checksum은 정확한 Git
commit에서 다시 생성해야 한다. GPU5와 서명 증거가 없는 번들은 데모 및 개발 검토용
`UNVERIFIED artifact-only` 결과이며 검증 릴리스가 아니다. v0.4.0 JSON, EXE 또는
테스트 결과는 v0.4.1 완료 증거로 사용할 수 없다.
