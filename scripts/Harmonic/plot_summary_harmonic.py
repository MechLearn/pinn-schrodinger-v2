# scripts/Harmonic/plot_summary.py
#
# Genera las 4 figuras del paper del oscilador armónico:
#   Fig 1 — Heatmap de error relativo de energía (análogo al heatmap de colapso del paper 1)
#   Fig 2 — Panel doble: E_rel vs n  +  L2 vs n
#   Fig 3 — Funciones de onda representativas (mejor y peor convergencia)
#   Fig 4 — Comparación directa con pozo infinito (si hay datos disponibles)

import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── Rutas ──────────────────────────────────────────────────────────────────────
BASE_HARMONIC  = "outputs/Harmonic/runs"
BASE_INFINITE  = "outputs/InfiniteWell/runs"
OUT_DIR        = "outputs/Harmonic/figs"

# ── Energías exactas ───────────────────────────────────────────────────────────
def exact_energy(n):
    return float(n) - 0.5


# ── Carga de datos ─────────────────────────────────────────────────────────────
def load_metrics(base):
    pattern = os.path.join(base, "*_seed*", "mode_*", "metrics.json")
    files   = sorted(glob.glob(pattern))
    rows    = []
    for fp in files:
        with open(fp) as f:
            m = json.load(f)
        n = int(m.get("n", -1))
        rows.append({
            "experiment": m.get("experiment_name", "exp"),
            "seed":       int(m.get("seed", -1)),
            "n":          n,
            "E_rel":      float(m.get("E_rel", np.nan)),
            "L2":         float(m.get("L2",    np.nan)),
            "integral":   float(m.get("integral", np.nan)),
            "E_learned":  float(m.get("E_learned", np.nan)),
            "E_exact":    float(m.get("E_exact",   exact_energy(n))),
            "collapse":   float(m.get("integral", 1.0)) < 0.2 and float(m.get("L2", 0.0)) > 0.5,
        })
    return pd.DataFrame(rows).sort_values(["experiment", "n", "seed"]).reset_index(drop=True)


# ── Fig 1 — Heatmap de E_rel mediano ──────────────────────────────────────────
def fig1_heatmap(df, out_dir):
    archs = sorted(df["experiment"].unique())
    ns    = sorted(df["n"].unique())

    pivot = df.groupby(["experiment", "n"])["E_rel"].median().unstack("n")
    pivot = pivot.reindex(index=archs, columns=ns)

    fig, ax = plt.subplots(figsize=(14, 2.5))
    data = pivot.values.astype(float)

    # Escala logarítmica para el color
    lognorm = mcolors.LogNorm(vmin=np.nanmin(data[data > 0]), vmax=np.nanmax(data))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", norm=lognorm)

    ax.set_xticks(range(len(ns)));     ax.set_xticklabels(ns, fontsize=8)
    ax.set_yticks(range(len(archs))); ax.set_yticklabels(
        [f"{a} ({128 if 'A1' in a else 256} neur.)" for a in archs], fontsize=9
    )
    ax.set_xlabel("Modo n", fontsize=10)
    ax.set_title("Error relativo de energía mediano $\\tilde{\\varepsilon}_n$ — Oscilador Armónico", fontsize=11)

    # Anotaciones
    for i, arch in enumerate(archs):
        for j, n in enumerate(ns):
            val = pivot.loc[arch, n]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if val > 0.3 else "black")

    plt.colorbar(im, ax=ax, label="$\\varepsilon_n$ mediano")
    plt.tight_layout()
    path = os.path.join(out_dir, "fig1_heatmap_Erel.png")
    plt.savefig(path, dpi=160, bbox_inches="tight"); plt.close()
    print(f"  ✓ {path}")


# ── Fig 2 — Panel doble: E_rel y L2 vs n ─────────────────────────────────────
def fig2_panels(df, out_dir):
    archs  = sorted(df["experiment"].unique())
    colors = {"A1": "#1f77b4", "A2": "#ff7f0e"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for arch in archs:
        sub     = df[df["experiment"] == arch]
        summary = sub.groupby("n").agg(
            E_rel_med  = ("E_rel", "median"),
            E_rel_std  = ("E_rel", "std"),
            L2_med     = ("L2",    "median"),
            L2_std     = ("L2",    "std"),
        ).reset_index()

        col   = colors.get(arch, "gray")
        label = f"{arch} ({128 if 'A1' in arch else 256} neur.)"

        # Panel a — E_rel
        axes[0].plot(summary["n"], summary["E_rel_med"], marker="o", color=col, label=label)
        axes[0].fill_between(
            summary["n"],
            summary["E_rel_med"] - summary["E_rel_std"],
            summary["E_rel_med"] + summary["E_rel_std"],
            alpha=0.15, color=col
        )

        # Panel b — L2
        axes[1].plot(summary["n"], summary["L2_med"], marker="s", color=col, label=label)
        axes[1].fill_between(
            summary["n"],
            summary["L2_med"] - summary["L2_std"],
            summary["L2_med"] + summary["L2_std"],
            alpha=0.15, color=col
        )

    for ax, ylabel, title in zip(
        axes,
        ["$\\tilde{\\varepsilon}_n$ (mediana)", "$\\tilde{L}^2_n$ (mediana)"],
        ["(a) Error relativo de energía vs modo $n$",
         "(b) Error $L^2$ de la función de onda vs modo $n$"]
    ):
        ax.set_xlabel("Modo $n$", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Oscilador Armónico — Deriva de energía sin colapso", fontsize=12)
    plt.tight_layout()
    path = os.path.join(out_dir, "fig2_Erel_L2_panels.png")
    plt.savefig(path, dpi=160, bbox_inches="tight"); plt.close()
    print(f"  ✓ {path}")


# ── Fig 3 — Funciones de onda representativas ─────────────────────────────────
def fig3_wavefunctions(out_dir):
    """
    Carga las imágenes mode.png de un modo bajo (n=1) y un modo alto (n=10)
    para A1_seed0 y las combina en una figura de dos paneles.
    """
    cases = [
        (BASE_HARMONIC, "A1", 0, 1,  "(a) $n=1$ (convergencia típica)"),
        (BASE_HARMONIC, "A1", 0, 10, "(b) $n=10$ (deriva de energía)"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, (base, arch, seed, n, label) in zip(axes, cases):
        # Buscar la carpeta correcta
        pattern = os.path.join(base, f"{arch}_seed{seed}", f"mode_{n}", "mode.png")
        matches = glob.glob(pattern)
        if matches:
            img = plt.imread(matches[0])
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(label, fontsize=10)
        else:
            ax.text(0.5, 0.5, f"No encontrado:\n{pattern}",
                    ha="center", va="center", transform=ax.transAxes, fontsize=8)
            ax.axis("off")

    plt.suptitle("Funciones de onda — Oscilador Armónico", fontsize=12)
    plt.tight_layout()
    path = os.path.join(out_dir, "fig3_wavefunctions.png")
    plt.savefig(path, dpi=160, bbox_inches="tight"); plt.close()
    print(f"  ✓ {path}")


# ── Fig 4 — Comparación Harmonic vs InfiniteWell ──────────────────────────────
def fig4_comparison(df_harm, out_dir):
    """
    Compara E_rel mediano entre oscilador armónico y pozo infinito (si hay datos).
    """
    try:
        df_inf = load_metrics(BASE_INFINITE)
        has_infinite = len(df_inf) > 0
    except Exception:
        has_infinite = False

    if not has_infinite:
        print("  ⚠ No hay datos del pozo infinito para comparar. Saltando Fig 4.")
        return

    fig, ax = plt.subplots(figsize=(10, 4))

    for df, label, color, marker in [
        (df_harm, "Oscilador Armónico (A1)", "#1f77b4", "o"),
        (df_inf,  "Pozo Infinito (A1)",      "#2ca02c", "s"),
    ]:
        sub = df[df["experiment"].str.contains("A1")]
        if len(sub) == 0:
            continue
        summary = sub.groupby("n")["E_rel"].median().reset_index()
        ax.plot(summary["n"], summary["E_rel"], marker=marker, label=label, color=color)

    ax.set_xlabel("Modo $n$", fontsize=10)
    ax.set_ylabel("$\\tilde{\\varepsilon}_n$ mediano", fontsize=10)
    ax.set_title("Comparación de error de energía: Pozo Infinito vs Oscilador Armónico (A1)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "fig4_comparison.png")
    plt.savefig(path, dpi=160, bbox_inches="tight"); plt.close()
    print(f"  ✓ {path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Cargando datos del oscilador armónico...")
    df = load_metrics(BASE_HARMONIC)

    if len(df) == 0:
        print("No se encontraron datos. Verifica que el sweep terminó.")
        return

    print(f"  {len(df)} corridas encontradas")
    print(f"  Arquitecturas: {sorted(df['experiment'].unique())}")
    print(f"  Modos: {sorted(df['n'].unique())}")

    print("\nGenerando figuras...")
    fig1_heatmap(df, OUT_DIR)
    fig2_panels(df, OUT_DIR)
    fig3_wavefunctions(OUT_DIR)
    fig4_comparison(df, OUT_DIR)

    print(f"\nTodas las figuras guardadas en: {OUT_DIR}")


if __name__ == "__main__":
    main()
