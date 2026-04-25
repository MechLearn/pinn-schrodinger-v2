import os
import json
import glob
from collections import defaultdict

import numpy as np
import pandas as pd


def is_collapse(m: dict, integral_thr=0.2, l2_thr=0.5) -> bool:
    """Criterio operativo de colapso (ajustable)."""
    integral = float(m.get("integral", np.nan))
    l2 = float(m.get("L2", np.nan))
    # Colapso si integral muy bajo Y L2 muy alto (solución ~0 o forma mala)
    return (integral < integral_thr) and (l2 > l2_thr)


def main():
    base = "outputs/runs"
    pattern = os.path.join(base, "*_seed*", "mode_*", "metrics.json")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No encontré metrics.json con patrón: {pattern}")
        return

    rows = []
    for fp in files:
        with open(fp, "r") as f:
            m = json.load(f)

        exp = m.get("experiment_name", "exp")
        seed = int(m.get("seed", -1))
        n = int(m.get("n", -1))

        rows.append({
            "experiment": exp,
            "seed": seed,
            "n": n,
            "E_rel": float(m.get("E_rel", np.nan)),
            "L2": float(m.get("L2", np.nan)),
            "integral": float(m.get("integral", np.nan)),
            "train_time_sec": float(m.get("train_time_sec", np.nan)),
            "collapse": is_collapse(m),
            "path": fp,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(["experiment", "n", "seed"]).reset_index(drop=True)

    # ---- Resumen por (experiment, n)
    def agg_bool(x):  # colapsos totales
        return int(np.sum(x))

    summary = (
        df.groupby(["experiment", "n"])
          .agg(
              runs=("seed", "count"),
              collapses=("collapse", agg_bool),
              collapse_rate=("collapse", "mean"),
              E_rel_mean=("E_rel", "mean"),
              E_rel_std=("E_rel", "std"),
              L2_mean=("L2", "mean"),
              L2_std=("L2", "std"),
              integral_mean=("integral", "mean"),
              integral_std=("integral", "std"),
              time_mean=("train_time_sec", "mean"),
              time_std=("train_time_sec", "std"),
          )
          .reset_index()
          .sort_values(["experiment", "n"])
    )

    # ---- Guardar CSVs
    os.makedirs("outputs/summary", exist_ok=True)
    df.to_csv("outputs/summary/all_runs.csv", index=False)
    summary.to_csv("outputs/summary/summary_by_n.csv", index=False)

    # ---- Imprimir tabla corta en consola (solo n=10..20 si aplica)
    print("\n=== SUMMARY (por n) ===")
    print(summary.to_string(index=False))

    # También: mostrar dónde colapsó, para inspección rápida
    collapsed = df[df["collapse"]].copy()
    if len(collapsed) > 0:
        print("\n=== Corridas colapsadas (para revisar) ===")
        print(collapsed[["experiment", "seed", "n", "integral", "L2", "E_rel", "path"]].to_string(index=False))
    else:
        print("\nNo detecté colapsos con el criterio actual. (Puedes ajustar thresholds)")

    print("\nGuardado:")
    print(" - outputs/summary/all_runs.csv")
    print(" - outputs/summary/summary_by_n.csv")


if __name__ == "__main__":
    main()