# src/pinn1d/Harmonic/train.py

import os
import math
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import hermite as hermite_poly

from .model import make_net
from .losses import compute_losses

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def exact_energy(n: int) -> float:
    return float(n) - 0.5


def exact_wavefunction(xs: np.ndarray, n: int) -> np.ndarray:
    k    = n - 1
    Hk   = hermite_poly(k)
    norm = 1.0 / np.sqrt(2**k * math.factorial(k) * np.sqrt(np.pi))
    return norm * Hk(xs.squeeze()) * np.exp(-0.5 * xs.squeeze()**2)


def run_one_mode_learnE(n: int, cfg: dict):
    base_dir = cfg.get("save_dir", "outputs/Harmonic/runs")
    exp_name = cfg.get("experiment_name", "harmonic")
    seed     = cfg.get("seed", 0)

    save_dir = os.path.join(base_dir, f"{exp_name}_seed{seed}", f"mode_{n}")
    os.makedirs(save_dir, exist_ok=True)

    print(f"Usando device: {DEVICE}")

    HIDDEN   = cfg["model"]["hidden"]
    USE_SINE = cfg["model"]["use_sine"]
    X_MAX    = float(cfg.get("x_max", 6.0))

    N_col    = max(1024, 2048 * n)
    EPOCHS   = 20000 if n >= 4 else (15000 if n == 3 else (10000 if n == 2 else 8000))
    LR0      = 3e-4  if n >= 4 else (5e-4 if n == 3 else (7e-4 if n == 2 else 1e-3))
    lam_hi, lam_lo = (300.0, 80.0) if n >= 3 else (40.0, 15.0 if n == 2 else 10.0)

    net    = make_net(n=n, hidden=HIDDEN, use_sine=USE_SINE).to(DEVICE)
    x_col  = torch.linspace(-X_MAX, X_MAX, N_col, dtype=torch.float32).reshape(-1, 1).to(DEVICE)

    E_ref      = exact_energy(n)
    alpha_init = float(E_ref + np.log(1.0 - np.exp(-E_ref) + 1e-8))
    alpha      = torch.tensor(alpha_init, dtype=torch.float32, device=DEVICE, requires_grad=True)

    params = list(net.parameters()) + [alpha]
    opt    = torch.optim.Adam(params, lr=LR0)

    def get_lr(ep):
        return LR0 * (0.1 + 0.9 * (1 - ep / EPOCHS))

    loss_total, loss_pde, loss_norm, E_hist, int_hist, epochs_logged = [], [], [], [], [], []
    lam = lam_hi

    for ep in range(1, EPOCHS + 1):
        if ep == EPOCHS * 2 // 3:
            lam = lam_lo
        for pg in opt.param_groups:
            pg["lr"] = get_lr(ep)

        opt.zero_grad()
        L, LPDE, Lnorm, integral, E = compute_losses(net, x_col, alpha, lam)
        L.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        opt.step()

        if ep <= 500 or ep % 200 == 0 or ep == EPOCHS:
            loss_total.append(float(L.detach()))
            loss_pde.append(float(LPDE.detach()))
            loss_norm.append(float(Lnorm.detach()))
            E_hist.append(float(E.detach()))
            int_hist.append(float(integral.detach()))
            epochs_logged.append(ep)

        if ep % max(1000, EPOCHS // 5) == 0 or ep == 1:
            print(
                f"n={n} ep={ep} "
                f"E={float(E.detach()):.6f} "
                f"L={float(L.detach()):.3e} "
                f"LPDE={float(LPDE.detach()):.3e} "
                f"Lnorm={float(Lnorm.detach()):.3e} "
                f"integral={float(integral.detach()):.6f} "
                f"lam={lam:.1f}"
            )

    net.eval()
    with torch.no_grad():
        xs       = torch.linspace(-X_MAX, X_MAX, 2000, dtype=torch.float32).reshape(-1, 1).to(DEVICE)
        psi_pred = net(xs).cpu().numpy().squeeze()

    xs_np     = np.linspace(-X_MAX, X_MAX, 2000, dtype=np.float32)
    psi_exact = exact_wavefunction(xs_np.reshape(-1, 1), n)
    sign      = np.sign(np.dot(psi_pred, psi_exact))
    psi_pred *= sign

    l2_err    = float(np.sqrt(np.trapezoid((psi_pred - psi_exact)**2, xs_np)))
    integ     = float(np.trapezoid(psi_pred**2, xs_np))
    E_learned = float(torch.nn.functional.softplus(alpha).item())
    E_exact   = exact_energy(n)
    E_rel     = float(abs(E_learned - E_exact) / (abs(E_exact) + 1e-12))

    plt.figure(figsize=(7, 4))
    plt.plot(xs_np, psi_pred,  label="PINN")
    plt.plot(xs_np, psi_exact, "--", label="Exacta")
    plt.title(f"n={n} | E={E_learned:.6f} | rel={E_rel:.2e} | L2={l2_err:.2e} | integral={integ:.4f}")
    plt.xlabel("x"); plt.ylabel("psi"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "mode.png"), dpi=150); plt.close()

    plt.figure(figsize=(7, 4))
    plt.semilogy(epochs_logged, loss_total, label="Total")
    plt.semilogy(epochs_logged, loss_pde,   label="PDE")
    plt.semilogy(epochs_logged, loss_norm,  label="Norm")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss.png"), dpi=150); plt.close()

    metrics = {
        "potential": "harmonic", "n": n,
        "experiment_name": exp_name, "seed": int(seed),
        "E_learned": E_learned, "E_exact": E_exact, "E_rel": E_rel,
        "L2": l2_err, "integral": integ,
        "final_loss": loss_total[-1], "final_pde": loss_pde[-1], "final_norm": loss_norm[-1],
        "epochs": EPOCHS, "N_col": N_col, "hidden": HIDDEN, "use_sine": USE_SINE,
        "x_max": X_MAX, "lam_hi": lam_hi, "lam_lo": lam_lo, "lr0": LR0,
        "device": str(DEVICE),
    }
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    np.savez(
        os.path.join(save_dir, "history.npz"),
        loss_total=np.array(loss_total, dtype=np.float64),
        loss_pde=np.array(loss_pde, dtype=np.float64),
        loss_norm=np.array(loss_norm, dtype=np.float64),
        E_hist=np.array(E_hist, dtype=np.float64),
        int_hist=np.array(int_hist, dtype=np.float64),
        epochs_logged=np.array(epochs_logged, dtype=np.int32),
    )

    return {
        "potential": "harmonic", "n": n, "save_dir": save_dir,
        "E_learned": E_learned, "E_exact": E_exact, "E_rel": E_rel,
        "L2": l2_err, "integral": integ,
        "mode_png": os.path.join(save_dir, "mode.png"),
        "loss_png": os.path.join(save_dir, "loss.png"),
        "metrics_json": os.path.join(save_dir, "metrics.json"),
        "history_npz": os.path.join(save_dir, "history.npz"),
    }
