"""Generate a deterministic local release dependency SBOM and notice file."""

from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from hashlib import sha256
from importlib import metadata
import json
import os
from pathlib import Path
import re
import tomllib
from typing import Iterable, Sequence
from uuid import UUID


_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*")


def _normalized_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).casefold()


def _requirement_name(value: str) -> str | None:
    matched = _NAME.match(value.strip())
    return matched.group(0) if matched else None


def _project_roots(pyproject: Path) -> set[str]:
    payload = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    project = payload.get("project")
    if not isinstance(project, dict):
        raise ValueError("pyproject project table is missing")
    requirements: list[object] = list(project.get("dependencies") or [])
    optional = project.get("optional-dependencies") or {}
    if isinstance(optional, dict):
        for values in optional.values():
            if isinstance(values, list):
                requirements.extend(values)
    roots = {
        name
        for value in requirements
        if isinstance(value, str) and (name := _requirement_name(value)) is not None
    }
    if not roots:
        raise ValueError("no runtime dependency roots were found")
    return roots


def _active_requirement_names(distribution: metadata.Distribution) -> set[str]:
    result: set[str] = set()
    for value in distribution.requires or ():
        name = _requirement_name(value)
        if name is None:
            continue
        try:
            from packaging.requirements import Requirement

            parsed = Requirement(value)
            if parsed.marker is not None and not any(
                parsed.marker.evaluate({"extra": extra}) for extra in ("", "gemma")
            ):
                continue
            name = parsed.name
        except (ImportError, ValueError):
            # Including a marker dependency is safer than silently omitting a
            # potentially shipped component when packaging is unavailable.
            pass
        result.add(name)
    return result


def _distribution_graph(
    roots: Iterable[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    pending = sorted({_normalized_name(name) for name in roots})
    seen: set[str] = set()
    components: list[dict[str, object]] = []
    edges: dict[str, set[str]] = {}
    while pending:
        requested = pending.pop(0)
        if requested in seen:
            continue
        try:
            distribution = metadata.distribution(requested)
        except metadata.PackageNotFoundError:
            components.append(
                {
                    "type": "library",
                    "name": requested,
                    "version": "NOT_INSTALLED",
                    "bom-ref": f"pkg:pypi/{requested}@NOT_INSTALLED",
                    "properties": [
                        {"name": "cogni:resolution", "value": "not-installed"}
                    ],
                }
            )
            edges[requested] = set()
            seen.add(requested)
            continue
        canonical = _normalized_name(distribution.metadata.get("Name", requested))
        if canonical in seen:
            continue
        version = distribution.version
        bom_ref = f"pkg:pypi/{canonical}@{version}"
        declared_license = (
            distribution.metadata.get("License-Expression")
            or distribution.metadata.get("License")
            or "UNKNOWN"
        ).strip()[:512]
        component: dict[str, object] = {
            "type": "library",
            "name": distribution.metadata.get("Name", canonical),
            "version": version,
            "bom-ref": bom_ref,
            "purl": bom_ref,
            "properties": [
                {"name": "cogni:declared_license", "value": declared_license},
                {
                    "name": "cogni:resolution",
                    "value": "installed-build-environment",
                },
            ],
        }
        homepage = distribution.metadata.get("Home-page")
        if homepage and homepage.startswith(("https://", "http://")):
            component["externalReferences"] = [
                {"type": "website", "url": homepage[:2048]}
            ]
        dependencies = {
            _normalized_name(name) for name in _active_requirement_names(distribution)
        }
        edges[canonical] = dependencies
        pending.extend(sorted(dependencies - seen - set(pending)))
        components.append(component)
        seen.add(canonical)
    components.sort(key=lambda item: str(item["bom-ref"]))
    by_name = {
        _normalized_name(str(item["name"])): str(item["bom-ref"]) for item in components
    }
    dependency_graph = [
        {
            "ref": by_name[name],
            "dependsOn": sorted(
                by_name[target] for target in targets if target in by_name
            ),
        }
        for name, targets in sorted(edges.items())
        if name in by_name
    ]
    return components, dependency_graph


def _timestamp() -> str:
    raw = os.environ.get("SOURCE_DATE_EPOCH")
    instant = (
        datetime.fromtimestamp(int(raw), tz=timezone.utc)
        if raw and raw.isdecimal()
        else datetime.now(timezone.utc)
    )
    return instant.isoformat().replace("+00:00", "Z")


def _artifact_components(paths: Iterable[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(paths, key=lambda item: item.name.casefold()):
        resolved = path.resolve(strict=True)
        digest = sha256(resolved.read_bytes()).hexdigest()
        rows.append(
            {
                "type": "file",
                "name": resolved.name,
                "bom-ref": f"file:{resolved.name}:{digest}",
                "hashes": [{"alg": "SHA-256", "content": digest}],
            }
        )
    return rows


def generate(
    *, pyproject: Path, project_version: str, artifacts: Sequence[Path]
) -> tuple[dict[str, object], str]:
    if not re.fullmatch(r"\d+\.\d+\.\d+", project_version):
        raise ValueError("project version is invalid")
    components, dependency_graph = _distribution_graph(_project_roots(pyproject))
    file_components = _artifact_components(artifacts)
    identity = sha256(
        json.dumps(
            [project_version, components, file_components],
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).digest()[:16]
    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": f"urn:uuid:{UUID(bytes=identity)}",
        "version": 1,
        "metadata": {
            "timestamp": _timestamp(),
            "component": {
                "type": "application",
                "name": "Cogni-OS",
                "version": project_version,
                "bom-ref": f"pkg:pypi/cogni-os@{project_version}",
            },
            "properties": [
                {
                    "name": "cogni:scope",
                    "value": "build-environment dependency closure and release artifacts",
                },
                {
                    "name": "cogni:signature_status",
                    "value": "unsigned-no-code-signing-certificate-provided",
                },
            ],
        },
        "components": components + file_components,
        "dependencies": dependency_graph,
    }
    notice_lines = [
        "# Cogni-OS third-party dependency inventory",
        "",
        "This inventory records the installed build-environment dependency closure. ",
        "It is not legal advice; redistribute only after reviewing each upstream license.",
        "",
        "| Package | Version | Declared license metadata |",
        "|---|---:|---|",
    ]
    for item in components:
        properties = {
            str(row["name"]): str(row["value"])
            for row in item.get("properties", [])
            if isinstance(row, dict) and "name" in row and "value" in row
        }
        license_text = properties.get("cogni:declared_license", "UNKNOWN").replace(
            "|", "\\|"
        )
        notice_lines.append(f"| {item['name']} | {item['version']} | {license_text} |")
    notice_lines.extend(
        [
            "",
            "Gemma model weights and AkasicDB source are not bundled by this launcher. ",
            "Review their licenses and provenance separately before distribution.",
            "",
        ]
    )
    return sbom, "\n".join(notice_lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = ArgumentParser()
    parser.add_argument("--pyproject", required=True)
    parser.add_argument("--project-version", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--notices", required=True)
    parser.add_argument("--artifact", action="append", default=[])
    args = parser.parse_args(argv)
    sbom, notices = generate(
        pyproject=Path(args.pyproject).resolve(strict=True),
        project_version=args.project_version,
        artifacts=[Path(value) for value in args.artifact],
    )
    output = Path(args.output).resolve()
    notice_path = Path(args.notices).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    notice_path.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(sbom, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    notice_path.write_text(notices, encoding="utf-8")
    print(f"sbom_components={len(sbom['components'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
