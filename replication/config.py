"""Load and validate the declarative paper asset manifest."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_ALLOWED_ASSET_KINDS = {"figure", "tex"}
_ALLOWED_ASSET_STATUSES = {"current", "pending"}
_GRAPHICS_RE = re.compile(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}")
_INPUT_RE = re.compile(r"\\input\{([^}]+)\}")


@dataclass(frozen=True)
class PaperManifest:
    """Validated paper manifest and its repository root."""

    root: Path
    path: Path
    payload: dict[str, Any]

    @property
    def assets(self) -> dict[str, dict[str, Any]]:
        return self.payload["assets"]

    @property
    def calculations(self) -> dict[str, dict[str, Any]]:
        return self.payload.get("calculations", {})


def _without_comments(text: str) -> str:
    """Remove unescaped TeX comments before looking for asset references."""
    lines = []
    for line in text.splitlines():
        match = re.search(r"(?<!\\)%", line)
        lines.append(line if match is None else line[: match.start()])
    return "\n".join(lines)


def manuscript_asset_references(root: Path) -> set[tuple[str, str]]:
    """Return `(TeX source, referenced path)` for generated paper assets."""
    references: set[tuple[str, str]] = set()
    for path in sorted((root / "tex").glob("*.tex")):
        text = _without_comments(path.read_text())
        source = path.relative_to(root).as_posix()
        references.update((source, match) for match in _GRAPHICS_RE.findall(text))
        references.update(
            (source, match)
            for match in _INPUT_RE.findall(text)
            if match.startswith("../output/") and "#" not in match
        )
    return references


def _validate(payload: dict[str, Any], root: Path) -> None:
    if payload.get("schema_version") != 1:
        raise ValueError("paper.yaml must have schema_version: 1")

    assets = payload.get("assets")
    if not isinstance(assets, dict) or not assets:
        raise ValueError("paper.yaml must declare a nonempty assets mapping")
    calculations = payload.get("calculations", {})
    if not isinstance(calculations, dict):
        raise ValueError("calculations must be a mapping")

    outputs: dict[str, str] = {}
    current_references: set[tuple[str, str]] = set()
    for asset_id, asset in assets.items():
        if not isinstance(asset, dict):
            raise ValueError(f"asset {asset_id!r} must be a mapping")
        if asset.get("kind") not in _ALLOWED_ASSET_KINDS:
            raise ValueError(f"asset {asset_id!r} has invalid kind")
        if asset.get("status") not in _ALLOWED_ASSET_STATUSES:
            raise ValueError(f"asset {asset_id!r} has invalid status")
        output = asset.get("output")
        if not isinstance(output, str) or not output:
            raise ValueError(f"asset {asset_id!r} must declare output")
        if output in outputs:
            raise ValueError(
                f"assets {outputs[output]!r} and {asset_id!r} share output {output!r}"
            )
        outputs[output] = asset_id

        calculation = asset.get("calculation")
        if calculation is not None and calculation not in calculations:
            raise ValueError(
                f"asset {asset_id!r} references unknown calculation {calculation!r}"
            )
        if asset["status"] == "current":
            try:
                current_references.add((asset["tex_source"], asset["tex_path"]))
            except KeyError as error:
                raise ValueError(
                    f"current asset {asset_id!r} must declare tex_source and tex_path"
                ) from error

    actual_references = manuscript_asset_references(root)
    missing = sorted(actual_references - current_references)
    stale = sorted(current_references - actual_references)
    if missing or stale:
        messages = []
        if missing:
            messages.append(f"undeclared manuscript assets: {missing}")
        if stale:
            messages.append(f"current assets absent from manuscript: {stale}")
        raise ValueError("; ".join(messages))

    controlled = payload.get("controlled_benchmark_inputs", [])
    if not isinstance(controlled, list):
        raise ValueError("controlled_benchmark_inputs must be a list")
    absent = [item for item in controlled if not (root / item).is_file()]
    if absent:
        raise ValueError(f"missing controlled benchmark inputs: {absent}")


def load_paper_manifest(path: str | Path = "paper.yaml") -> PaperManifest:
    """Load `paper.yaml`, validate its asset graph, and return it."""
    manifest_path = Path(path).resolve()
    payload = yaml.safe_load(manifest_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("paper.yaml root must be a mapping")
    root = manifest_path.parent
    _validate(payload, root)
    return PaperManifest(root=root, path=manifest_path, payload=payload)
