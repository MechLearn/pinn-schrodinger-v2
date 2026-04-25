from pinn1d.InfiniteWell.config import load_config
from pinn1d.InfiniteWell.train import run_one_mode_learnE
import tensorflow as tf
import sys

if __name__ == "__main__":
    cfg = load_config("configs/base.yaml")

    # Pedir al usuario interactivamente
    print("=== PINN Schrödinger 1D ===")
    n_min = int(input("Modo inicial (n_min): "))
    n_max = int(input("Modo final  (n_max): "))

    print(f"\nEntrenando modos del {n_min} al {n_max}...")

    # semilla desde config
    tf.keras.utils.set_random_seed(int(cfg.get("seed", 0)))

    for n in range(n_min, n_max + 1):
        print(f"\n===== Running mode n={n} =====")
        run_one_mode_learnE(n, cfg)

    print(f"\n✅ Completado! Modos {n_min} a {n_max} entrenados.")