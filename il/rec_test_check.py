"""Gate before the full recording: the short test recording must have samples of every type, the speed-history columns
filled and plausible (signed forward speed: small negatives = rolling back), and speed_m025 must equal the speed of the same vehicle's sample 0.25 s earlier."""
import sys

import pandas as pd

d = pd.read_csv(sys.argv[1] + "/samples.csv")
t = pd.read_csv(sys.argv[1] + "/timing.csv")
cols = [f"speed_m{h * 25:03d}" for h in (1, 2, 3, 4)]
print(f"{len(d):,} samples from {len(t)} scenarios; by aug type:", d.groupby("aug").size().to_dict())
errs = []
miss = [c for c in cols if c not in d]
if miss:
    errs.append(f"missing columns {miss}")
else:
    print(d[["speed"] + cols].describe().round(2).to_string())
    if d[cols].isna().any().any():
        errs.append("NaN in speed history")
    if (d[cols] < -2.0).any().any() or (d[cols] > 30).any().any():
        errs.append("speed history out of range")
    prev = d[["scenario", "track", "k", "speed"]].assign(k=d.k + 1)  # sample at k-1 -> its speed is speed_m025 at k
    m = d.merge(prev, on=["scenario", "track", "k"], suffixes=("", "_prev"))
    bad = (m.speed_m025 - m.speed_prev).abs() > 1e-3
    print(f"speed_m025 vs the speed of the sample 0.25 s earlier: {len(m):,} pairs, {bad.sum()} mismatches")
    if len(m) == 0 or bad.any():
        errs.append("speed history does not match earlier samples")
if t.samples.sum() == 0 or d.aug.nunique() < 2:
    errs.append("too few sample types")
print("RESULT:", "ok" if not errs else "; ".join(errs))
sys.exit(1 if errs else 0)
