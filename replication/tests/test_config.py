from __future__ import annotations

import unittest
from pathlib import Path

from replication.config import load_paper_manifest, manuscript_asset_references


ROOT = Path(__file__).resolve().parents[2]


class PaperManifestTests(unittest.TestCase):
    def test_manifest_matches_current_manuscript_assets(self) -> None:
        manifest = load_paper_manifest(ROOT / "paper.yaml")
        declared = {
            (asset["tex_source"], asset["tex_path"])
            for asset in manifest.assets.values()
            if asset["status"] == "current"
        }
        self.assertEqual(declared, manuscript_asset_references(ROOT))

    def test_both_foa_summaries_are_approved_paper_assets(self) -> None:
        manifest = load_paper_manifest(ROOT / "paper.yaml")
        for asset_id in ("foa_principal_summary", "foa_fixed_action_summary"):
            asset = manifest.assets[asset_id]
            self.assertEqual(asset["kind"], "figure")
            self.assertEqual(asset["status"], "pending")
            self.assertNotIn("mock", asset["output"])

    def test_only_controlled_benchmark_outputs_are_tracked_inputs(self) -> None:
        manifest = load_paper_manifest(ROOT / "paper.yaml")
        self.assertEqual(
            manifest.payload["controlled_benchmark_inputs"],
            ["output/timing_results.csv", "output/machine_specs.txt"],
        )


if __name__ == "__main__":
    unittest.main()
