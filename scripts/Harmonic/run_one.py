# scripts/Harmonic/run_one.py

from pinn1d.Harmonic.config import load_config
from pinn1d.Harmonic.train import run_one_mode_learnE
import tensorflow as tf
import sys

if __name__ == "__main__":
    cfg = load_config("configs/Harmonic/base.yaml")

    if len(sys.argv) < 2:
        print("Uso: python scripts/Harmonic/run_one.py <n>")
        raise SystemExit(1)

    n      = int(sys.argv[1])
    seed_n = int(cfg.get("seed", 0)) * 100 + n
    tf.keras.utils.set_random_seed(seed_n)

    print(f"=== Oscilador Armónico · n={n} · seed={seed_n} ===")
    run_one_mode_learnE(n, cfg)