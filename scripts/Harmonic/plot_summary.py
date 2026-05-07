# scripts/Harmonic/plot_summary.py
#
# Figuras para el paper del oscilador armónico:
#
#   Fig 1 — Heatmap de L2 por modo y arquitectura
#            (muestra claramente los dos regímenes: L2<0.2 vs L2≈1.41)
#
#   Fig 2 — Panel doble: L2 vs n  +  integral vs n
#            (L2≈√2 es la firma de convergencia al modo incorrecto)
#
#   Fig 3 — Funciones de onda representativas:
#            n=1 (modo correcto), n=3 (primer modo incorrecto), n=10
#
#   Fig 4 — Comparación directa pozo infinito vs oscilador armónico:
#            colapso (ψ→0) vs convergencia al modo incorrecto (L2≈√2)

import os
import json
import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── Rutas ──────────────────────────────────────────────────────────────────────
BASE_HARMONIC = "outputs/Harmonic/runs"
BASE_INFINITE = "outputs/InfiniteWell/runs"
OUT_DIR       = "outputs/Harmonic/figs"
SQRT2         = math.sqrt(2)


# ── Carga de datos ─────────────────────────────────────────────────────────────
def load_metrics(base):
    pattern = os.path.join(base, "*_seed*", "mode_*", "metrics.json")
    files   = sorted(glob.glob(pattern))
    rows    = []
    for fp in files:
        with open(fp) as f:
            m = json.load(f)
        n = int(m.get("n", -1))
        integral = float(m.get("integral", 1.0))
        l2       = float(m.get("L2", 0.0))
        rows.append({
            "experiment": m.get("experiment_name", "exp"),
            "seed":       int(m.get("seed", -1)),
            "n":          n,
            "E_rel":      float(m.get("E_rel",     np.nan)),
            "L2":         l2,
            "integral":   integral,
            "E_learned":  float(m.get("E_learned", np.nan)),
            "E_exact":    float(m.get("E_exact",   float(n) - 0.5)),
            # modo_incorrecto: L2 cerca de √2 (función ortogonal a la exacta)
            "wrong_mode": l2 > 1.0,
            # colapso clásico: integral muy baja
            "collapse":   integral < 0.2 and l2 > 0.5,
        })
    return pd.DataFrame(rows).sort_values(["experiment", "n", "seed"]).reset_index(drop=True)


# ── Fig 1 — Heatmap de L2 mediano ─────────────────────────────────────────────
def fig1_heatmap_L2(df, out_dir):
    archs = sorted(df["experiment"].unique())
    ns    = sorted(df["n"].unique())

    pivot = df.groupby(["experiment", "n"])["L2"].median().unstack("n")
    pivot = pivot.reindex(index=archs, columns=ns)

    fig, ax = plt.subplots(figsize=(14, 2.8))
    data = pivot.values.astype(float)

    # Colormap: blanco=correcto (L2≈0), rojo=modo incorrecto (L2≈√2)
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1.5)

    ax.set_xticks(range(len(ns)))
    ax.set_xticklabels(ns, fontsize=8)
    ax.set_yticks(range(len(archs)))
    ax.set_yticklabels(
        [f"{a} ({128 if 'A1' in a else 256} neur.)" for a in archs],
        fontsize=9
    )
    ax.set_xlabel("Modo $n$", fontsize=10)
    ax.set_title(
        "Error $L^2$ mediano de la función de onda — Oscilador Armónico\n"
        r"($L^2 \approx \sqrt{2}$ indica convergencia al modo incorrecto)",
        fontsize=10
    )

    # Línea de referencia √2
    for i in range(len(archs)):
        for j, n in enumerate(ns):
            val = pivot.iloc[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if val > 0.8 else "black")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("$L^2$ mediano", fontsize=9)
    # Marcar √2 en la colorbar
    cbar.ax.axhline(SQRT2, color="blue", linewidth=1.5, linestyle="--")
    cbar.ax.text(1.1, SQRT2/1.5, f"$\\sqrt{{2}}$={SQRT2:.2f}", fontsize=8,
                 color="blue", transform=cbar.ax.transAxes, va="center")

    plt.tight_layout()
    path = os.path.join(out_dir, "fig1_heatmap_L2.png")
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


# ── Fig 2 — Panel doble: L2 y integral vs n ───────────────────────────────────
def fig2_L2_integral(df, out_dir):
    archs  = sorted(df["experiment"].unique())
    colors = {"A1": "#1f77b4", "A2": "#ff7f0e"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for arch in archs:
        sub = df[df["experiment"] == arch]
        summary = sub.groupby("n").agg(
            L2_med       = ("L2",       "median"),
            L2_std       = ("L2",       "std"),
            int_med      = ("integral", "median"),
            int_std      = ("integral", "std"),
        ).reset_index()

        col   = colors.get(arch, "gray")
        label = f"{arch} ({128 if 'A1' in arch else 256} neur.)"

        # Panel a — L2
        axes[0].plot(summary["n"], summary["L2_med"],
                     marker="o", color=col, label=label)
        axes[0].fill_between(
            summary["n"],
            summary["L2_med"] - summary["L2_std"],
            summary["L2_med"] + summary["L2_std"],
            alpha=0.15, color=col
        )

        # Panel b — integral
        axes[1].plot(summary["n"], summary["int_med"],
                     marker="s", color=col, label=label)
        axes[1].fill_between(
            summary["n"],
            summary["int_med"] - summary["int_std"],
            summary["int_med"] + summary["int_std"],
            alpha=0.15, color=col
        )

    # Línea de referencia √2 en panel a
    axes[0].axhline(SQRT2, color="gray", linestyle="--", linewidth=1,
                    label=f"$\\sqrt{{2}}$ = {SQRT2:.3f}")
    axes[0].axhline(0.0, color="green", linestyle=":", linewidth=1,
                    label="Perfecto ($L^2=0$)")

    # Línea de referencia en panel b
    axes[1].axhline(1.0, color="gray", linestyle="--", linewidth=1,
                    label="Normalización exacta")

    # Marcar n=3 — primer modo incorrecto
    for ax in axes:
        ax.axvline(3, color="red", linestyle=":", linewidth=1, alpha=0.5)
        ax.text(3.1, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 0.01,
                "n=3", color="red", fontsize=8, alpha=0.7)

    titles = [
        "(a) Error $L^2$ de la función de onda vs modo $n$",
        "(b) Integral de normalización vs modo $n$"
    ]
    ylabels = [
        "$\\tilde{L}^2_n$ (mediana)",
        "$\\int \\hat{\\psi}^2\\,dx$ (mediana)"
    ]
    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.set_xlabel("Modo $n$", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Oscilador Armónico — Convergencia al modo incorrecto ($L^2 \\approx \\sqrt{2}$) a partir de $n=3$",
        fontsize=11
    )
    plt.tight_layout()
    path = os.path.join(out_dir, "fig2_L2_integral_panels.png")
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


# ── Fig 3 — Funciones de onda representativas ─────────────────────────────────
def fig3_wavefunctions(out_dir):
    """
    Tres paneles:
    (a) n=1 — modo correcto (L2≈0.08)
    (b) n=3 — primer modo incorrecto (L2≈1.41)
    (c) n=10 — modo incorrecto en región alta (L2≈1.41)
    """
    cases = [
        (BASE_HARMONIC, "A1", 0, 1,  "(a) $n=1$ — modo correcto"),
        (BASE_HARMONIC, "A1", 0, 3,  "(b) $n=3$ — primer modo incorrecto"),
        (BASE_HARMONIC, "A1", 0, 10, "(c) $n=10$ — modo incorrecto"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, (base, arch, seed, n, label) in zip(axes, cases):
        pattern = os.path.join(base, f"{arch}_seed{seed}", f"mode_{n}", "mode.png")
        matches = glob.glob(pattern)
        if matches:
            img = plt.imread(matches[0])
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(label, fontsize=10)
        else:
            ax.text(0.5, 0.5, f"No encontrado:\n{pattern}",
                    ha="center", va="center",
                    transform=ax.transAxes, fontsize=8)
            ax.axis("off")

    plt.suptitle(
        "Funciones de onda — Oscilador Armónico (A1, seed=0)\n"
        "Para $n \\geq 3$ la red converge a un modo diferente al pedido",
        fontsize=11
    )
    plt.tight_layout()
    path = os.path.join(out_dir, "fig3_wavefunctions.png")
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


# ── Fig 4 — Comparación Harmonic vs InfiniteWell ──────────────────────────────
def fig4_comparison(df_harm, out_dir):
    """
    Compara los dos modos de fallo:
    - Pozo infinito: colapso (integral → 0)
    - Oscilador armónico: modo incorrecto (L2 → √2)
    Métrica común: L2 mediano vs n para A1
    """
    try:
        df_inf      = load_metrics(BASE_INFINITE)
        has_infinite = len(df_inf) > 0
    except Exception:
        has_infinite = False

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Panel a — L2 mediano
    harm_A1 = df_harm[df_harm["experiment"].str.contains("A1")]
    summary_harm = harm_A1.groupby("n")["L2"].median().reset_index()
    axes[0].plot(summary_harm["n"], summary_harm["L2"],
                 marker="o", color="#1f77b4",
                 label="Oscilador Armónico (A1)")

    if has_infinite:
        inf_A1 = df_inf[df_inf["experiment"].str.contains("A1")]
        summary_inf = inf_A1.groupby("n")["L2"].median().reset_index()
        axes[0].plot(summary_inf["n"], summary_inf["L2"],
                     marker="s", color="#2ca02c",
                     label="Pozo Infinito (A1)")

    axes[0].axhline(SQRT2, color="gray", linestyle="--", linewidth=1,
                    label=f"$\\sqrt{{2}}={SQRT2:.3f}$ (modos ortogonales)")
    axes[0].set_xlabel("Modo $n$", fontsize=10)
    axes[0].set_ylabel("$L^2$ mediano", fontsize=10)
    axes[0].set_title("(a) Error $L^2$ — dos modos de fallo", fontsize=10)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Panel b — integral mediana (colapso vs normalización correcta)
    summary_harm_int = harm_A1.groupby("n")["integral"].median().reset_index()
    axes[1].plot(summary_harm_int["n"], summary_harm_int["integral"],
                 marker="o", color="#1f77b4",
                 label="Oscilador Armónico (A1)")

    if has_infinite:
        summary_inf_int = inf_A1.groupby("n")["integral"].median().reset_index()
        axes[1].plot(summary_inf_int["n"], summary_inf_int["integral"],
                     marker="s", color="#2ca02c",
                     label="Pozo Infinito (A1)")

    axes[1].axhline(1.0, color="gray", linestyle="--", linewidth=1,
                    label="Normalización exacta")
    axes[1].set_xlabel("Modo $n$", fontsize=10)
    axes[1].set_ylabel("$\\int \\hat{\\psi}^2\\,dx$ mediana", fontsize=10)
    axes[1].set_title("(b) Normalización — colapso vs modo incorrecto", fontsize=10)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([-0.05, 1.1])

    plt.suptitle(
        "Comparación de modos de fallo: Pozo Infinito vs Oscilador Armónico\n"
        "Pozo infinito: colapso ($\\int\\psi^2 \\to 0$) — "
        "Oscilador: modo incorrecto ($L^2 \\to \\sqrt{2}$)",
        fontsize=11
    )
    plt.tight_layout()
    path = os.path.join(out_dir, "fig4_comparison.png")
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Cargando datos del oscilador armónico...")
    df = load_metrics(BASE_HARMONIC)

    if len(df) == 0:
        print("No se encontraron datos.")
        return

    print(f"  {len(df)} corridas encontradas")
    print(f"  Arquitecturas: {sorted(df['experiment'].unique())}")
    print(f"  Modos: {sorted(df['n'].unique())}")

    # Diagnóstico rápido
    wrong = df[df["wrong_mode"]]
    print(f"\n  Corridas con modo incorrecto (L2>1.0): {len(wrong)}/{len(df)}")
    print(f"  Primer modo incorrecto: n={wrong['n'].min() if len(wrong)>0 else 'N/A'}")

    print("\nGenerando figuras...")
    fig1_heatmap_L2(df, OUT_DIR)
    fig2_L2_integral(df, OUT_DIR)
    fig3_wavefunctions(OUT_DIR)
    fig4_comparison(df, OUT_DIR)

    print(f"\nTodas las figuras guardadas en: {OUT_DIR}")


if __name__ == "__main__":
    main()