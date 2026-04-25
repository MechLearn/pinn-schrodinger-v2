from pinn1d.InfiniteWell.config import load_config
from pinn1d.InfiniteWell.train import run_one_mode_learnE
import tensorflow as tf
import sys

if __name__ == "__main__":
    cfg = load_config("configs/base.yaml")

    if len(sys.argv) < 2:
        print("Uso: python scripts/run_one.py <n>")
        raise SystemExit(1)

    n = int(sys.argv[1])

    # semilla desde config
    tf.keras.utils.set_random_seed(int(cfg.get("seed", 0)))

    run_one_mode_learnE(n, cfg)