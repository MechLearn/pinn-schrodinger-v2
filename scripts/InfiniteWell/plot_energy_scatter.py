import os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    in_csv = "outputs/summary/all_runs.csv"
    if not os.path.exists(in_csv):
        raise FileNotFoundError(
            f"No existe: {in_csv}. Ejecuta primero scripts/summarize_metrics.py"
        )

    df = pd.read_csv(in_csv)
    df = df[df["experiment"] == "A1"].copy()
    df = df.sort_values(["n", "seed"])

    out_dir = "outputs/summary/figs"
    os.makedirs(out_dir, exist_ok=True)

    # Scatter: colapsados vs no colapsados
    df_ok = df[df["collapse"] == False]
    df_bad = df[df["collapse"] == True]

    plt.figure()
    if len(df_ok) > 0:
        plt.scatter(df_ok["n"], df_ok["E_rel"], label="No colapsa", marker="o")
    if len(df_bad) > 0:
        plt.scatter(df_bad["n"], df_bad["E_rel"], label="Colapsa", marker="x")

    plt.yscale("log")  # E_rel suele variar en órdenes de magnitud
    plt.xlabel("n")
    plt.ylabel("E_rel (log scale)")
    plt.title("A1: Relative energy error vs mode n (colored by collapse)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    outpath = os.path.join(out_dir, "A1_Erel_scatter_by_collapse.png")
    plt.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close()

    print("Listo. Figura guardada en:")
    print(f" - {outpath}")


if __name__ == "__main__":
    main()