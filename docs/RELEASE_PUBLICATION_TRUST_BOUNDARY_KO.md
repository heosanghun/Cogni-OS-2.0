# 릴리스 게시 신뢰 경계

`PublishRelease`는 현재 `EXTERNAL_BLOCKER`이다. 저장소의
`config/release-toolchain-policy.json`은 v2 폐쇄 스키마지만 의도적으로 `unconfigured`이며,
이 상태에서는 PATH 탐색이나 Git/Python 실행 전에 게시가 실패한다. `approved` 정책을 시험하더라도
실행 중인 PowerShell의 정규 경로와 SHA-256, `-NoProfile -NonInteractive` 실행을 먼저 확인한 뒤,
전체 빌드 클로저와 오프라인 wheelhouse를 실제 격리 설치·실행에 결속하는 외부 runner가 없으면
역시 Python/Git 실행 전에 실패한다. 일반 artifact-only 빌드는 개발 편의를 위해 허용되지만
결과는 항상 `UNVERIFIED`이다.

`VERIFIED` 게시를 승인하려면 저장소 소유 사용자와 분리된 다음 외부 실행 경계가 먼저 필요하다.

- root/administrator가 소유하고 일반 빌드 사용자가 수정할 수 없는 PowerShell, Python, Git 및
  Python 표준 라이브러리·빌드 백엔드·Git libexec 전체의 해시 고정 인벤토리
- 정책에 고정된 build-closure manifest와 offline-wheelhouse manifest만 설치·가져오는
  `-NoProfile -NonInteractive` clean-host, 네트워크 차단 저권한 격리 worker
- 해당 worker가 생성한 immutable SUBJECT, signed release attestation, detached effective
  acceptance bundle의 동일 바이트 검증과 게시

실행 파일 하나의 SHA-256만 승인하는 정책은 전체 런타임과 빌드 백엔드를 식별하지 못하므로
신뢰 루트로 간주하지 않는다. 이 외부 조건이 충족되기 전에는 `status=approved`로 변경하거나
`VERIFIED`라고 표기해서는 안 된다.

최종 조립 단계는 payload 파일을 read-lock한 바이트에서 `SHA256SUMS.txt`를 만들고, checksum을
포함한 전체 staging 인벤토리를 다시 잠가 비교한 다음 이동 후 같은 인벤토리를 재검증한다. 다만
Windows에서는 디렉터리 이동 직전에 파일 잠금을 해제해야 하므로, 그 짧은 unlock/rename 구간에서
다른 프로세스의 쓰기 자체를 배제하는 책임은 위의 보호된 격리 runner에 있다. 게시용 수동 PDF도
운영자가 SHA-256을 제공해야 하며, 검증된 private snapshot handle에서만 번들로 복사된다.
