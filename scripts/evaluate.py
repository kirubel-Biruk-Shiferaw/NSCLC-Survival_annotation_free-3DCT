#!/usr/bin/env python3
"""
Evaluate trained models: compute metrics, bootstrap CIs, Wilcoxon tests.

Usage
-----
# Evaluate all experiments whose OOF files exist:
python scripts/evaluate.py --out-dir outputs/

# Only bootstrap CIs for key experiments:
python scripts/evaluate.py --out-dir outputs/ --bootstrap-only

Environment variables
---------------------
NSCLC_OUTPUT_DIR : same as set for training (default: ./outputs)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from nsclc_survival.utils.config import load_config
from nsclc_survival.data.clinical import load_labels
from nsclc_survival.evaluation.metrics import (
    bootstrap_cindex,
    wilcoxon_compare,
    integrated_brier_score_oof,
    format_results_table,
)


def load_oof(oof_dir: Path, exp: str) -> dict | None:
    path = oof_dir / f"{exp}_oof.npz"
    if not path.exists():
        return None
    d = np.load(str(path))
    return {
        "eta":           d["eta"],
        "surv":          d["surv"],
        "event":         d["event"],
        "fold_cindices": d["fold_cindices"].tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate NSCLC survival models")
    parser.add_argument("--config",         default="configs/default.yaml")
    parser.add_argument("--out-dir",        default=None,
                        help="Override output directory from config")
    parser.add_argument("--bootstrap-only", action="store_true")
    parser.add_argument("--no-ibs",         action="store_true",
                        help="Skip Integrated Brier Score (slow)")
    args = parser.parse_args()

    cfg     = load_config(args.config)
    out_dir = Path(args.out_dir or cfg.output.output_dir)
    oof_dir = out_dir / "oof_predictions"
    csv_path = Path(cfg.data.artifact_dir) / cfg.data.csv_filename

    df = load_labels(csv_path)

    ALL_EXPS = [
        "E1","E2","E3","E4","E5","E6","E7","E8","E9",
        "E10","E11","E12","E13","E14","E15",
        "E16a","E16b","E17a","E17b","E17c","E17d",
    ]
    KEY_EXPS = cfg.evaluation.key_exps_for_ci

    # ── Load OOF predictions ──────────────────────────────────────────────
    oof = {}
    for exp in ALL_EXPS:
        result = load_oof(oof_dir, exp)
        if result is not None:
            oof[exp] = result

    if not oof:
        print(f"No OOF predictions found in {oof_dir}.  Run train.py first.")
        sys.exit(1)

    print(f"Found OOF predictions for: {list(oof.keys())}")

    # ── Bootstrap CIs ─────────────────────────────────────────────────────
    bootstrap_res = {}
    for exp in KEY_EXPS:
        if exp not in oof:
            continue
        print(f"  Bootstrap CI for {exp}…", end=" ", flush=True)
        pt, lo, hi = bootstrap_cindex(
            oof[exp]["surv"], oof[exp]["eta"], oof[exp]["event"],
            n_resamples=cfg.evaluation.bootstrap_n_resamples,
            rng_seed=cfg.training.seed,
        )
        bootstrap_res[exp] = (pt, lo, hi)
        print(f"  {pt:.4f}  ({lo:.4f}, {hi:.4f})")

    # ── Wilcoxon tests vs E10 ─────────────────────────────────────────────
    wilcoxon_res = {}
    if "E10" in oof:
        for exp in oof:
            if exp == "E10":
                continue
            wr = wilcoxon_compare(
                oof[exp]["fold_cindices"],
                oof["E10"]["fold_cindices"],
                name_a=exp, name_b="E10",
            )
            wilcoxon_res[exp] = wr
            if wr["valid"]:
                print(f"  {exp} vs E10: W={wr['statistic']:.1f}  p={wr['p_value']:.4f}")
            else:
                print(f"  {exp} vs E10: {wr['note']}")

    # ── IBS ───────────────────────────────────────────────────────────────
    if not args.no_ibs:
        print("\nIntegrated Brier Scores (OOF):")
        ibs_res = {}
        for exp in KEY_EXPS:
            if exp not in oof:
                continue
            try:
                ibs = integrated_brier_score_oof(
                    train_surv=oof[exp]["surv"],    # approximate: use full OOF as proxy
                    train_event=oof[exp]["event"],
                    test_surv=oof[exp]["surv"],
                    test_event=oof[exp]["event"],
                    test_risk=oof[exp]["eta"],
                )
                ibs_res[exp] = ibs
                print(f"  {exp}: IBS={ibs:.4f}")
            except Exception as e:
                print(f"  {exp}: IBS failed — {e}")

    # ── Results table ──────────────────────────────────────────────────────
    results_for_table = {
        exp: {
            "oof_cindex":    float(np.concordance_index_ipcw_stub(oof[exp])),
            "fold_cindices": oof[exp]["fold_cindices"],
        }
        for exp in oof
    }
    # Compute OOF C-index inline
    from lifelines.statistics import concordance_index as ci_fn
    results_for_table = {
        exp: {
            "oof_cindex": float(ci_fn(
                oof[exp]["surv"], -oof[exp]["eta"], oof[exp]["event"])),
            "fold_cindices": oof[exp]["fold_cindices"],
        }
        for exp in oof
    }

    table = format_results_table(results_for_table, bootstrap_res, wilcoxon_res)
    print("\n" + table)

    # ── Save evaluation results ───────────────────────────────────────────
    eval_out = {
        "oof_cindices":  {exp: r["oof_cindex"] for exp, r in results_for_table.items()},
        "bootstrap_cis": {exp: list(v) for exp, v in bootstrap_res.items()},
        "wilcoxon":      {exp: {k: v for k, v in wr.items() if k != "note"}
                          for exp, wr in wilcoxon_res.items() if wr["valid"]},
    }
    eval_path = out_dir / "evaluation_results.json"
    eval_path.write_text(json.dumps(eval_out, indent=2, default=str))
    print(f"\nEvaluation results saved to {eval_path}")

    # Save Markdown table
    table_path = out_dir / "results_table.md"
    table_path.write_text(table + "\n")
    print(f"Markdown table saved to {table_path}")


if __name__ == "__main__":
    main()
