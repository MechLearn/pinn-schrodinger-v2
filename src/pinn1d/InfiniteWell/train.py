# src/pinn1d/train.py

import os
import math
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from .model import make_net
from .losses import compute_losses


def run_one_mode_learnE(n: int, cfg: dict):
    # -------------------------
    # Output directory (from cfg)
    # -------------------------
    base_dir = cfg.get("save_dir", "outputs/runs")
    exp_name = cfg.get("experiment_name", "exp")
    seed = cfg.get("seed", 0)

    save_dir = os.path.join(base_dir, f"{exp_name}_seed{seed}", f"mode_{n}")
    os.makedirs(save_dir, exist_ok=True)

    # -------------------------
    # Heuristics 
    # -------------------------
    HIDDEN = cfg["model"]["hidden"]
    USE_SINE = cfg["model"]["use_sine"]
    N_col    = max(1024, 2048 * n)
    EPOCHS   = 15000 if n >= 4 else (9000 if n == 3 else (6000 if n == 2 else 4000))
    LR0      = 3e-4  if n >= 4 else (5e-4 if n == 3 else (7e-4 if n == 2 else 1e-3))
    lam_hi, lam_lo = (300.0, 80.0) if n >= 3 else (40.0, 15.0 if n == 2 else 10.0)

    # -------------------------
    # Model + collocation points
    # -------------------------
    net = make_net(n=n, hidden=HIDDEN, use_sine=USE_SINE)

    x_col = np.linspace(0, 1, N_col, dtype=np.float32).reshape(-1, 1)
    x_batch = tf.constant(x_col)

    # -------------------------
    # Trainable energy parameter
    # -------------------------
    E_init = np.float32((n * math.pi) ** 2)
    alpha = tf.Variable(E_init, dtype=tf.float32)

    # -------------------------
    # Optimizer
    # -------------------------
    lr_sched = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=LR0,
        decay_steps=EPOCHS,
        end_learning_rate=LR0 * 0.1,
        power=1.0
    )
    opt = tf.keras.optimizers.Adam(learning_rate=lr_sched, clipnorm=1.0)

    vars_all = net.trainable_variables + [alpha]

    # -------------------------
    # History
    # -------------------------
    loss_total, loss_pde, loss_norm, E_hist, int_hist = [], [], [], [], []

    # -------------------------
    # Training loop
    # -------------------------
    for ep in range(1, EPOCHS + 1):
        lam = lam_hi if ep < EPOCHS // 3 else lam_lo

        with tf.GradientTape() as tape:
            L, LPDE, Lnorm, integral, E = compute_losses(net, x_batch, alpha, lam)

        grads = tape.gradient(L, vars_all)
        opt.apply_gradients(zip(grads, vars_all))

        # history
        loss_total.append(float(L))
        loss_pde.append(float(LPDE))
        loss_norm.append(float(Lnorm))
        E_hist.append(float(E))
        int_hist.append(float(integral))

        if ep % max(1000, EPOCHS // 5) == 0 or ep == 1:
            print(
                f"n={n} ep={ep} "
                f"E={E.numpy():.6f} "
                f"L={L.numpy():.3e} "
                f"LPDE={LPDE.numpy():.3e} "
                f"Lnorm={Lnorm.numpy():.3e} "
                f"∫ψ²≈{integral.numpy():.6f} "
                f"λ={lam}"
            )

    # -------------------------
    # Evaluation + saving artifacts
    # -------------------------
    xs = np.linspace(0, 1, 2000, dtype=np.float32).reshape(-1, 1)
    psi_pred = net(xs).numpy().squeeze()
    psi_exact = (np.sqrt(2.0) * np.sin(n * math.pi * xs)).squeeze()

    # align sign
    sign = np.sign(np.dot(psi_pred, psi_exact))
    psi_pred *= sign

    # metrics
    l2_err = float(np.sqrt(np.mean((psi_pred - psi_exact) ** 2)))
    integ = float(np.trapz(psi_pred ** 2, xs.squeeze()))

    E_learned = float(tf.nn.softplus(alpha).numpy())
    E_exact = float((n * math.pi) ** 2)
    E_rel = float(abs(E_learned - E_exact) / (abs(E_exact) + 1e-12))

    # mode plot
    plt.figure(figsize=(7, 4))
    plt.plot(xs.squeeze(), psi_pred, label="PINN")
    plt.plot(xs.squeeze(), psi_exact, "--", label="Exacta")
    plt.title(
        f"n={n} | E={E_learned:.6f} | rel={E_rel:.2e} | "
        f"L2={l2_err:.2e} | ∫ψ²={integ:.4f}"
    )
    plt.xlabel("x")
    plt.ylabel("ψ")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "mode.png"), dpi=150)
    plt.close()

    # loss plot
    plt.figure(figsize=(7, 4))
    plt.semilogy(loss_total, label="Total")
    plt.semilogy(loss_pde, label="PDE")
    plt.semilogy(loss_norm, label="Norm")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss.png"), dpi=150)
    plt.close()

    # metrics.json
    metrics = {
        "n": n,
        "experiment_name": cfg.get("experiment_name", "A1"),
        "seed": int(cfg.get("seed", 0)),
        "E_learned": E_learned,
        "E_exact": E_exact,
        "E_rel": E_rel,
        "L2": l2_err,
        "integral": integ,
        "final_loss": loss_total[-1] if loss_total else None,
        "final_pde": loss_pde[-1] if loss_pde else None,
        "final_norm": loss_norm[-1] if loss_norm else None,
        "epochs": EPOCHS,
        "N_col": N_col,
        "hidden": HIDDEN,
        "use_sine": USE_SINE,
        "lam_hi": lam_hi,
        "lam_lo": lam_lo,
        "lr0": LR0,
    }
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # optional: save history arrays (for later plots/sweep analysis)
    np.savez(
        os.path.join(save_dir, "history.npz"),
        loss_total=np.array(loss_total, dtype=np.float64),
        loss_pde=np.array(loss_pde, dtype=np.float64),
        loss_norm=np.array(loss_norm, dtype=np.float64),
        E_hist=np.array(E_hist, dtype=np.float64),
        int_hist=np.array(int_hist, dtype=np.float64),
    )

    return {
        "n": n,
        "save_dir": save_dir,
        "E_learned": E_learned,
        "E_exact": E_exact,
        "E_rel": E_rel,
        "L2": l2_err,
        "integral": integ,
        "mode_png": os.path.join(save_dir, "mode.png"),
        "loss_png": os.path.join(save_dir, "loss.png"),
        "metrics_json": os.path.join(save_dir, "metrics.json"),
        "history_npz": os.path.join(save_dir, "history.npz"),
    }