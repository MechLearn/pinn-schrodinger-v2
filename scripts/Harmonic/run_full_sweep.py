import os
from pinn1d.config import load_config
from pinn1d.Harmonic.train import run_one_mode_learnE
import torch

CONFIGS = [
    "configs/Harmonic/A1.yaml",
    "configs/Harmonic/A2.yaml",
]
SEEDS = [0, 1, 2, 3, 4]
N_MIN = 1
N_MAX = 20

if __name__ == "__main__":
    for config_path in CONFIGS:
        cfg      = load_config(config_path)
        exp_name = cfg.get("experiment_name")
        hidden   = cfg["model"]["hidden"]
        base_dir = cfg.get("save_dir", "outputs/Harmonic/runs")

        print(f"\n{'='*50}")
        print(f"Arquitectura: {exp_name} ({hidden} neuronas)")
        print(f"{'='*50}")

        for seed in SEEDS:
            cfg["seed"] = seed
            print(f"\n--- Seed {seed} ---")

            for n in range(N_MIN, N_MAX + 1):
                # Saltar si ya existe metrics.json
                save_dir = os.path.join(base_dir, f"{exp_name}_seed{seed}", f"mode_{n}")
                metrics_path = os.path.join(save_dir, "metrics.json")
                if os.path.exists(metrics_path):
                    print(f"  n={n} ya completado, saltando...")
                    continue

                seed_n = seed * 100 + n
                torch.manual_seed(seed_n)
                print(f"\n===== n={n} seed={seed_n} =====")
                run_one_mode_learnE(n, cfg)

    print("\nCompletado.")
