import os
import json
import glob
import numpy as np
import pandas as pd


def exact_energy(n):
    return float(n) - 0.5


def is_collapse(m, integral_thr=0.2, l2_thr=0.5):
    integral = float(m.get("integral", np.nan))
    l2 = float(m.get("L2", np.nan))
    return (integral < integral_thr) and (l2 > l2_thr)


def main():
    base    = "outputs/Harmonic/runs"
    pattern = os.path.join(base, "*_seed*", "mode_*", "metrics.json")
    files   = sorted(glob.glob(pattern))

    if not files:
        print(f"No encontré metrics.json con patrón: {pattern}")
        return

    rows = []
    for fp in files:
        with open(fp, "r") as f:
            m = json.load(f)

        n = int(m.get("n", -1))
        rows.append({
            "experiment":  m.get("experiment_name", "exp"),
            "seed":        int(m.get("seed", -1)),
            "n":           n,
            "E_exact":     exact_energy(n),
            "E_learned":   float(m.get("E_learned", np.nan)),
            "E_rel":       float(m.get("E_rel", np.nan)),
            "L2":          float(m.get("L2", np.nan)),
            "integral":    float(m.get("integral", np.nan)),
            "collapse":    is_collapse(m),
            "path":        fp,
        })

    df      = pd.DataFrame(rows)
    df      = df.sort_values(["experiment", "n", "seed"]).reset_index(drop=True)

    summary = (
        df.groupby(["experiment", "n"])
          .agg(
              runs          = ("seed", "count"),
              collapses     = ("collapse", "sum"),
              collapse_rate = ("collapse", "mean"),
              E_rel_mean    = ("E_rel", "mean"),
              E_rel_std     = ("E_rel", "std"),
              L2_mean       = ("L2", "mean"),
              L2_std        = ("L2", "std"),
              integral_mean = ("integral", "mean"),
              integral_std  = ("integral", "std"),
          )
          .reset_index()
          .sort_values(["experiment", "n"])
    )

    os.makedirs("outputs/Harmonic/summary", exist_ok=True)
    df.to_csv("outputs/Harmonic/summary/all_runs.csv", index=False)
    summary.to_csv("outputs/Harmonic/summary/summary_by_n.csv", index=False)

    print("\n=== SUMMARY OSCILADOR ARMONICO (por n) ===")
    print(summary.to_string(index=False))

    collapsed = df[df["collapse"]]
    if len(collapsed) > 0:
        print("\n=== Corridas colapsadas ===")
        print(collapsed[["experiment", "seed", "n", "integral", "L2", "E_rel"]].to_string(index=False))
    else:
        print("\nNo se detectaron colapsos.")

    print("\nGuardado en outputs/Harmonic/summary/")


if __name__ == "__main__":
    main()
