"""Command-line entry point for paper reproduction."""

from __future__ import annotations

import argparse

from .config import load_paper_manifest


def _check(manifest_path: str) -> None:
    manifest = load_paper_manifest(manifest_path)
    current = sum(asset["status"] == "current" for asset in manifest.assets.values())
    pending = sum(asset["status"] == "pending" for asset in manifest.assets.values())
    print(
        f"Validated {manifest.path.name}: {current} current assets, "
        f"{pending} approved pending assets."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reproducible paper assets")
    parser.add_argument("command", choices=["check"])
    parser.add_argument("--manifest", default="paper.yaml")
    args = parser.parse_args()
    if args.command == "check":
        _check(args.manifest)


if __name__ == "__main__":
    main()
