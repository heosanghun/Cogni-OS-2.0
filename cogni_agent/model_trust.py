"""Single trust root for the audited local Gemma 4 E4B-it snapshot."""

from __future__ import annotations

from cogni_os.artifacts import (
    VerifiedArtifactSet,
    verify_closed_world_artifact_layout,
)


TRUSTED_GEMMA4_E4B_IT_REVISION = "a4c2d58be94dda072b918d9db64ee85c8ed34e3f"
TRUSTED_GEMMA4_E4B_IT_DIGESTS: tuple[tuple[str, str], ...] = (
    (
        "chat_template.jinja",
        "2f1b4d75d067bae3fe44e676721c7f077d243bc007156cb9c2f8b5836613d082",
    ),
    (
        "config.json",
        "33b10c02df3c2e8536cf323d29d53262aaa2f4d11dbe19bc729373fbe90295d4",
    ),
    (
        "generation_config.json",
        "d4226bbe3117d2d253ba4609720ba82c6c4ce4627a9a6ae05387c78983ac03de",
    ),
    (
        "model.safetensors",
        "cfbd3d2f1cd71bd471c37fe2bf8546d5028d41e5736f64e1ca6c6b8893125503",
    ),
    (
        "processor_config.json",
        "32bdf45d2ad4cc29a0822ddd157a182de76644f0419a6228d151495256e9813c",
    ),
    (
        "tokenizer.json",
        "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
    ),
    (
        "tokenizer_config.json",
        "90c3a3ba5bf53818383a58e1a776cbcacd2a038d4812eaa373e1522f2d06f3df",
    ),
)
TRUSTED_GEMMA4_E4B_IT_BENIGN_FILES = ("README.md",)
TRUSTED_GEMMA4_E4B_IT_BENIGN_DIRECTORIES = (".cache",)


def require_instruction_tuned_e4b(verified: VerifiedArtifactSet) -> None:
    """Bind a caller to the audited upstream identity and complete digest set."""

    identity = verified.identity
    if identity is None:
        raise ValueError("interactive Gemma startup requires model identity metadata")
    if (
        identity.family.casefold() != "gemma4"
        or identity.variant.casefold() != "e4b"
        or identity.role != "instruction_tuned"
        or identity.source != "google/gemma-4-E4B-it"
        or identity.revision != TRUSTED_GEMMA4_E4B_IT_REVISION
    ):
        raise ValueError("interactive Gemma startup requires google/gemma-4-E4B-it")
    if verified.digests != TRUSTED_GEMMA4_E4B_IT_DIGESTS:
        raise ValueError(
            "interactive Gemma startup rejected an untrusted E4B-it fingerprint"
        )


def verify_instruction_tuned_e4b_snapshot(
    verified: VerifiedArtifactSet,
) -> None:
    """Verify the pinned file digests and its closed-world loader directory."""

    require_instruction_tuned_e4b(verified)
    verify_closed_world_artifact_layout(
        verified,
        allowed_unmanifested_files=TRUSTED_GEMMA4_E4B_IT_BENIGN_FILES,
        allowed_unmanifested_directories=(TRUSTED_GEMMA4_E4B_IT_BENIGN_DIRECTORIES),
    )


__all__ = [
    "TRUSTED_GEMMA4_E4B_IT_DIGESTS",
    "TRUSTED_GEMMA4_E4B_IT_REVISION",
    "require_instruction_tuned_e4b",
    "verify_instruction_tuned_e4b_snapshot",
]
