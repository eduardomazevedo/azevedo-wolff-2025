from __future__ import annotations

import unittest
from pathlib import Path

from replication.config import (
    foa_summary_rows,
    load_paper_manifest,
    manuscript_asset_references,
)


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

    def test_foa_summaries_share_the_declared_paper_order(self) -> None:
        principal = foa_summary_rows("principal", ROOT / "paper.yaml")
        fixed = foa_summary_rows("fixed_action", ROOT / "paper.yaml")
        self.assertEqual(len(principal), 31)
        self.assertEqual(len(fixed), 34)
        self.assertEqual(
            [row for row in fixed if row[2] not in {None, *[item[2] for item in principal]}],
            [
                ("data", "Low", "gaussian_fixed_actions_log__intended_action-70"),
                ("data", "Near monopsony", "gaussian_fixed_actions_log__intended_action-130"),
            ],
        )
        self.assertIn(("header", "Student-t (empty safe region)", None), principal)

    def test_only_controlled_benchmark_outputs_are_tracked_inputs(self) -> None:
        manifest = load_paper_manifest(ROOT / "paper.yaml")
        self.assertEqual(
            manifest.payload["controlled_benchmark_inputs"],
            ["output/timing_results.csv", "output/machine_specs.txt"],
        )


if __name__ == "__main__":
    unittest.main()
