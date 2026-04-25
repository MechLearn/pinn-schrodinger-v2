# src/pinn1d/Harmonic/train.py

import os
import math
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.special import hermite as hermite_poly

from .model import make_net
from .losses import compute_losses


def exact_energy(n: int) -> float:
    """
    Energía exacta del oscilador armónico (ℏ=m=ω=1).
    Eₙ = n - 0.5  con n = 1, 2, 3, ...
    """
    return float(n) - 0.5


def exact_wavefunction(xs: np.ndarray, n: int) -> np.ndarray:
    """
    Eigenfunción exacta normalizada.
    ψₙ(x) = (2^(n-1) (n-1)! √π)^(-1/2) · H_{n-1}(x) · exp(-x²/2)
    """
    k    = n - 1
    Hk   = hermite_poly(k)
    norm = 1.0 / np.sqrt(2**k * math.factorial(k) * np.sqrt(np.pi))
    return norm * Hk(xs.squeeze()) * np.exp(-0.5 * xs.squeeze()**2)


def run_one_mode_learnE(n: int, cfg: dict):
    # -------------------------
    # Directorio de salida
    # -------------------------
    base_dir = cfg.get("save_dir", "outputs/runs_harmonic")
    exp_name = cfg.get("experiment_name", "harmonic")
    seed     = cfg.get("seed", 0)

    save_dir = os.path.join(base_dir, f"{exp_name}_seed{seed}", f"mode_{n}")
    os.makedirs(save_dir, exist_ok=True)

    # -------------------------
    # Hiperparámetros
    # -------------------------
    HIDDEN   = cfg["model"]["hidden"]
    USE_SINE = cfg["model"]["use_sine"]
    X_MAX    = float(cfg.get("x_max", 6.0))

    N_col = max(2048, 4096 * n)
    EPOCHS = 20000 if n >= 4 else (15000 if n == 3 else (10000 if n == 2 else 8000))
    LR0 = 3e-4 if n >= 4 else (5e-4 if n == 3 else (7e-4 if n == 2 else 1e-3))
    lam_hi, lam_lo = (300.0, 80.0) if n >= 3 else (40.0, 15.0 if n == 2 else 10.0)
    # -------------------------
    # Modelo + puntos de colación
    # -------------------------
    net = make_net(n=n, hidden=HIDDEN, use_sine=USE_SINE, x_max=X_MAX)

    x_col   = np.linspace(-X_MAX, X_MAX, N_col, dtype=np.float32).reshape(-1, 1)
    x_batch = tf.constant(x_col)

    # -------------------------
    # Energía entrenable
    # Inicializar cerca del valor exacto
    # -------------------------
    E_ref      = exact_energy(n)
    # inv_softplus estable: log(exp(y) - 1) para y > 0.5, directo para y pequeño
    alpha_init = np.float32(E_ref + np.log(1.0 - np.exp(-E_ref) + 1e-8))
    alpha      = tf.Variable(alpha_init, dtype=tf.float32)
    # -------------------------
    # Optimizador
    # -------------------------
    lr_sched = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=LR0,
        decay_steps=EPOCHS,
        end_learning_rate=LR0 * 0.1,
        power=1.0
    )
    opt      = tf.keras.optimizers.Adam(learning_rate=lr_sched, clipnorm=1.0)
    vars_all = net.trainable_variables + [alpha]

    # -------------------------
    # train_step compilado
    # -------------------------
    lam_var = tf.Variable(lam_hi, dtype=tf.float32, trainable=False)

    @tf.function
    def train_step(x_batch):
        with tf.GradientTape() as tape:
            L, LPDE, Lnorm, integral, E = compute_losses(
                net, x_batch, alpha, lam_var
            )
        grads = tape.gradient(L, vars_all)
        opt.apply_gradients(zip(grads, vars_all))
        return L, LPDE, Lnorm, integral, E

    # -------------------------
    # Historia
    # -------------------------
    loss_total, loss_pde, loss_norm, E_hist, int_hist, epochs_logged = [], [], [], [], [], []

    # -------------------------
    # Loop de entrenamiento
    # -------------------------
    for ep in range(1, EPOCHS + 1):

        if ep == EPOCHS * 2 // 3:   # cambiar a 2/3 en lugar de 1/3
            lam_var.assign(lam_lo)

        L, LPDE, Lnorm, integral, E = train_step(x_batch)

        if ep <= 500 or ep % 200 == 0 or ep == EPOCHS:
            loss_total.append(float(L))
            loss_pde.append(float(LPDE))
            loss_norm.append(float(Lnorm))
            E_hist.append(float(E))
            int_hist.append(float(integral))
            epochs_logged.append(ep)

        if ep % max(1000, EPOCHS // 5) == 0 or ep == 1:
            print(
                f"n={n} ep={ep} "
                f"E={float(E):.6f} "
                f"L={float(L):.3e} "
                f"LPDE={float(LPDE):.3e} "
                f"Lnorm={float(Lnorm):.3e} "
                f"∫ψ²≈{float(integral):.6f} "
                f"λ={float(lam_var):.1f}"
            )

    # -------------------------
    # Evaluación
    # -------------------------
    xs        = np.linspace(-X_MAX, X_MAX, 2000, dtype=np.float32).reshape(-1, 1)
    psi_pred  = net(xs).numpy().squeeze()
    psi_exact = exact_wavefunction(xs, n)

    sign      = np.sign(np.dot(psi_pred, psi_exact))
    psi_pred *= sign

    l2_err  = float(np.sqrt(np.trapz((psi_pred - psi_exact)**2, xs.squeeze())))
    integ   = float(np.trapz(psi_pred**2, xs.squeeze()))

    E_learned = float(tf.nn.softplus(alpha).numpy())
    E_exact   = exact_energy(n)
    E_rel     = float(abs(E_learned - E_exact) / (abs(E_exact) + 1e-12))

    # plot modo
    plt.figure(figsize=(7, 4))
    plt.plot(xs.squeeze(), psi_pred,  label="PINN")
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

    # plot loss
    plt.figure(figsize=(7, 4))
    plt.semilogy(epochs_logged, loss_total, label="Total")
    plt.semilogy(epochs_logged, loss_pde,   label="PDE")
    plt.semilogy(epochs_logged, loss_norm,  label="Norm")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss.png"), dpi=150)
    plt.close()

    # metrics.json
    metrics = {
        "potential":       "harmonic",
        "n":               n,
        "experiment_name": cfg.get("experiment_name", "harmonic"),
        "seed":            int(cfg.get("seed", 0)),
        "E_learned":       E_learned,
        "E_exact":         E_exact,
        "E_rel":           E_rel,
        "L2":              l2_err,
        "integral":        integ,
        "final_loss":      loss_total[-1] if loss_total else None,
        "final_pde":       loss_pde[-1]   if loss_pde   else None,
        "final_norm":      loss_norm[-1]  if loss_norm  else None,
        "epochs":          EPOCHS,
        "N_col":           N_col,
        "hidden":          HIDDEN,
        "use_sine":        USE_SINE,
        "x_max":           X_MAX,
        "lam_hi":          lam_hi,
        "lam_lo":          lam_lo,
        "lr0":             LR0,
    }
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    np.savez(
        os.path.join(save_dir, "history.npz"),
        loss_total    = np.array(loss_total,    dtype=np.float64),
        loss_pde      = np.array(loss_pde,      dtype=np.float64),
        loss_norm     = np.array(loss_norm,     dtype=np.float64),
        E_hist        = np.array(E_hist,        dtype=np.float64),
        int_hist      = np.array(int_hist,      dtype=np.float64),
        epochs_logged = np.array(epochs_logged, dtype=np.int32),
    )

    return {
        "potential":    "harmonic",
        "n":            n,
        "save_dir":     save_dir,
        "E_learned":    E_learned,
        "E_exact":      E_exact,
        "E_rel":        E_rel,
        "L2":           l2_err,
        "integral":     integ,
        "mode_png":     os.path.join(save_dir, "mode.png"),
        "loss_png":     os.path.join(save_dir, "loss.png"),
        "metrics_json": os.path.join(save_dir, "metrics.json"),
        "history_npz":  os.path.join(save_dir, "history.npz"),
    }