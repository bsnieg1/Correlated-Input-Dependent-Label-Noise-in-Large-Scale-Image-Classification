"""Runs many configs x seeds sequentially, skipping runs that already have a summary.json.

    python -m src.run_grid --pattern "mnist_s*" --seeds 0 1 2
    python -m src.run_grid --pattern "mnist_abl_*" --seeds 0 1 2

Safe to interrupt and restart. Failures are logged and the grid continues.
"""
import argparse
import time
import traceback
from pathlib import Path

import yaml

from src.train import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="mnist_*")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--configs", default="configs")
    args = parser.parse_args()

    configs = sorted(Path(args.configs).glob(f"{args.pattern}.yaml"))
    jobs = []
    for path in configs:
        cfg = yaml.safe_load(path.read_text())
        for seed in args.seeds:
            done = Path(cfg.get("output_dir", "results")) / cfg["experiment"] / f"seed{seed}" / "summary.json"
            if not done.exists():
                jobs.append((path, cfg, seed))
    print(f"{len(configs)} configs, {len(jobs)} runs to do")
    t_start = time.time()
    for k, (path, cfg, seed) in enumerate(jobs, 1):
        t0 = time.time()
        print(f"\n=== [{k}/{len(jobs)}] {cfg['experiment']} seed {seed}", flush=True)
        try:
            train(cfg, seed)
        except Exception:
            traceback.print_exc()
        elapsed = time.time() - t_start
        print(f"=== done in {time.time() - t0:.0f}s; eta {elapsed / k * (len(jobs) - k) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
