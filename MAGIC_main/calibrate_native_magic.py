"""Build the detector-side native MAGIC evaluation cache.

TRACE is stored as multiple graphs while its metadata uses concatenated node
IDs.  Run this command once before TRACE attacks so every attack reuses the
same clean operating threshold and original MAGIC evaluation population.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MAGIC_ROOT = ROOT / "MAGIC_main"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(MAGIC_ROOT) not in sys.path:
    sys.path.insert(0, str(MAGIC_ROOT))

from DL_detector import DLDetector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("trace",), default="trace")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    detector = DLDetector(
        dataset_name=args.dataset, device=args.device, top_k=1
    )
    cache = detector.build_native_evaluation_cache(force=args.force)
    metrics = cache["initial_metrics"]
    print("Native MAGIC TRACE calibration completed")
    print(f"evaluation_size={cache['evaluation_size']}")
    print(f"malicious_count={cache['malicious_count']}")
    print(f"threshold={cache['threshold']:.12g}")
    for key in ("auc", "f1", "precision", "recall"):
        print(f"{key.upper()}={metrics[key]:.12g}")
    print(
        f"TN={metrics['tn']} FN={metrics['fn']} "
        f"TP={metrics['tp']} FP={metrics['fp']}"
    )


if __name__ == "__main__":
    main()
