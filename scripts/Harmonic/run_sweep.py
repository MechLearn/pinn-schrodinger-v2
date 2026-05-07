from pinn1d.config import load_config
from pinn1d.Harmonic.train import run_one_mode_learnE
import torch

if __name__ == "__main__":
    cfg = load_config("configs/Harmonic/base.yaml")

    print("=== PINN Oscilador Armonico 1D ===")
    n_min = int(input("Modo inicial (n_min): "))
    n_max = int(input("Modo final  (n_max): "))

    base_seed = int(cfg.get("seed", 0))
    print(f"\nEntrenando modos {n_min}-{n_max} seed base={base_seed}")

    for n in range(n_min, n_max + 1):
        seed_n = base_seed * 100 + n
        torch.manual_seed(seed_n)
        print(f"\n===== Modo n={n} seed={seed_n} =====")
        run_one_mode_learnE(n, cfg)

    print(f"\nCompletado. Modos {n_min}-{n_max} entrenados.")
