from pinn1d.InfiniteWell.config import load_config
from pinn1d.InfiniteWell.train import run_one_mode_learnE
import torch
import sys

if __name__ == "__main__":
    cfg = load_config("configs/InfiniteWell/base.yaml")

    if len(sys.argv) < 2:
        print("Uso: python scripts/InfiniteWell/run_one.py <n>")
        raise SystemExit(1)

    n      = int(sys.argv[1])
    seed_n = int(cfg.get("seed", 0)) * 100 + n
    torch.manual_seed(seed_n)

    print(f"=== Pozo Infinito · n={n} · seed={seed_n} ===")
    run_one_mode_learnE(n, cfg)
