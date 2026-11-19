"""Dataset statistics: CVS class balance and inter-expert agreement (Fleiss' kappa)."""
import argparse
import json
from pathlib import Path

import numpy as np

from choleguard.data.endoscapes import EndoscapesCVSDataset
from choleguard.metrics import CRITERIA, fleiss_kappa


def main(root, out):
    stats = {}
    for split in ("train", "val", "test"):
        ds = EndoscapesCVSDataset(root, split)
        votes = np.array([r[2] for r in ds.rows if r[2] is not None])      # (N, 3 raters, 3 criteria)
        mean = votes.mean(1)
        stats[split] = dict(
            frames=len(ds.rows), videos=len({r[0] for r in ds.rows}),
            positive_rate={c: float((mean[:, j] >= 0.5).mean()) for j, c in enumerate(CRITERIA)},
            cvs_reached_rate=float((mean >= 0.5).all(1).mean()),
            unanimous_rate={c: float(np.isin(votes[:, :, j].sum(1), [0, 3]).mean()) for j, c in enumerate(CRITERIA)},
            fleiss_kappa={c: fleiss_kappa(votes[:, :, j]) for j, c in enumerate(CRITERIA)},
        )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="docs/results/data_stats.json")
    a = ap.parse_args()
    main(a.root, a.out)
