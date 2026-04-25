import os
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def plot_line(df, x, y, title, ylabel, outpath):
    plt.figure()
    plt.plot(df[x].values, df[y].values, marker="o")
    plt.xlabel(x)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close()


def main():
    in_csv = "outputs/summary/summary_by_n.csv"
    if not os.path.exists(in_csv):
        raise FileNotFoundError(f"No existe: {in_csv}. Ejecuta primero summarize_metrics.py")

    df = pd.read_csv(in_csv)

    # Filtrar experimento A1 (por si luego agregas A2)
    df = df[df["experiment"] == "A1"].copy()
    df = df.sort_values("n")

    out_dir = "outputs/summary/figs"
    ensure_dir(out_dir)

    plot_line(
        df, "n", "collapse_rate",
        title="A1: Collapse rate vs mode n",
        ylabel="collapse_rate",
        outpath=os.path.join(out_dir, "A1_collapse_rate_vs_n.png"),
    )

    plot_line(
        df, "n", "L2_mean",
        title="A1: Mean L2 error vs mode n",
        ylabel="L2_mean",
        outpath=os.path.join(out_dir, "A1_L2_mean_vs_n.png"),
    )

    plot_line(
        df, "n", "integral_mean",
        title="A1: Mean normalization integral vs mode n",
        ylabel="integral_mean",
        outpath=os.path.join(out_dir, "A1_integral_mean_vs_n.png"),
    )

    print("Listo. Figuras guardadas en:")
    print(f" - {out_dir}/A1_collapse_rate_vs_n.png")
    print(f" - {out_dir}/A1_L2_mean_vs_n.png")
    print(f" - {out_dir}/A1_integral_mean_vs_n.png")


if __name__ == "__main__":
    main()