# Self-Harness 운영자 E2E 증거 절차

이 절차는 제품 UI 기능이 아니다. 기본 CogniBoard 런타임은 계속 `proposal_only`이며,
서명 키·자동 승인·자동 승격·UI apply endpoint를 포함하지 않는다.

## 1. 신뢰 경계

`LinuxOciSandboxRunner`의 일반 CPU integration smoke는 production attestation이 아니다.
실제 경계의 독립 평가자는 다음 항목을 포함한 canonical
`cogni.self_harness.runner_attestation.v1` JSON을 제품 외부에서 서명해야 한다.

- exact runner configuration evidence SHA-256
- 로드된 `kernel_sandbox.py` SHA-256
- container engine 절대 경로와 SHA-256
- exact OCI image digest, daemon socket, `runc`
- regression/health argv SHA-256 allowlist
- kernel/network/host filesystem/ephemeral workspace production 경계 판정
- attestor ID, pinned public-key SHA-256, TTL, 1회 nonce

서명은 별도 파일의 raw 64-byte Ed25519 signature이다. 제품 저장소와 실행 프로세스에는
private signing key를 두지 않는다. `load_externally_attested_runner`는 statement·signature·raw
32-byte public key의 symlink/reparse/크기/정규형을 확인하고, 실제 로드된 runner와 모든 필드를
대조한 뒤 nonce ledger에 O_EXCL로 1회 소비한다. 서명·TTL·source·image·engine·command 중 하나라도
다르면 runner를 만들지 않는다.

## 2. 운영자 E2E 단계

운영자는 외부 서명이 준비된 `PromotionMode.ATTESTED` 서비스에만
`OperatorSelfHarnessE2E`를 주입한다.

1. 실제 실패 evidence와 같은 signature의 서로 다른 candidate replacement가 3개 이상 존재한다.
2. immutable snapshot regression의 `CandidateEvaluationV1`이 zero-exit pass이고 아직 유효하다.
3. `prepare`가 failure/candidate/evaluation/runner/source digest를 첫 append-only event로 고정한다.
4. 별도 외부 `HumanApprovalV1` 서명을 받은 후 `promote`를 호출한다.
5. atomic promotion 뒤 health가 실패하면 before bytes를 복원하고 terminal restore event를 남긴다.
6. health가 통과하면 committed journal record와 active source digest를 두 번째 event로 고정한다.
7. 그 committed record를 정확히 바인딩한 별도 `RollbackAuthorizationV1` 서명을 받는다.
8. `rollback`이 before bytes를 복원하고 health를 실행한다. 성공 시 byte-identical digest와 전체
   source-surface digest가 최초 상태와 같은지 확인한다. 실패 시 committed bytes를 재적용한다.

각 event는 이전 event SHA-256을 포함하고 O_EXCL로 생성된다. ledger를 재시작해도 nonce replay,
sequence 누락, canonical JSON 변조, signature scope 변경을 fail-closed로 검출한다.

## 3. 읽기 전용 검증

```text
python scripts/validate_self_harness_e2e.py \
  --evidence-dir <operator-evidence-directory> \
  --run-id <32-hex-run-id> \
  --approval-public-key <raw-32-byte-public-key> \
  --approval-public-key-sha256 <sha256> \
  --approver-id <allowlisted-operator-id>
```

CLI는 source mutation, runner 실행, 서명을 수행하지 않는다. promotion과 rollback 두 외부 서명,
journal before/after, health command, event chain, 최종 byte/source identity만 검증한다. 출력의
`production_attestation_reverified=false`는 의도된 보수적 표시다. 이 CLI는 runner statement의
독립 평가를 다시 수행하지 않으므로, 실제 statement·signature·원시 runner evidence는 별도
감사 묶음으로 보존해야 한다.

## 4. 현재 완료 판정

코드와 CPU 회귀만으로 ID 75를 완료 처리하지 않는다. 현재 상태는 `PARTIAL`이다. 독립 평가자가
발행한 실제 production runner statement와 current production boundary에서 생성된 전체 원시
E2E chain이 release scope에 결합될 때까지 CogniBoard UI와 Runtime Fact-book은
`proposal_only`를 유지한다.
