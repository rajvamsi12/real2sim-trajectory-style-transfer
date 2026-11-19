"""5 scenarios per behaviour class for the TTC showcase (2026-11-13): test split first, then val, then train."""
import contextlib, io, json, sys
import numpy as np
sys.path.insert(0, "il")
with contextlib.redirect_stdout(io.StringIO()):
    import closed_loop as CL
rng = np.random.default_rng(0)
scens = CL.test_scenarios(True)
order = {"test": 0, "val": 1, "train": 2}
pick = {}
for b in sorted({s["behaviour"] for s in scens}):
    c = [s for s in scens if s["behaviour"] == b]
    c = sorted(c, key=lambda s: (order[s["split"]], rng.random()))
    pick[b] = [(s["id"], s["split"]) for s in c[:5]]
json.dump(pick, open("Scenariosssss/IL/showcase_ttc_pick.json", "w"), indent=1)
for b, v in pick.items():
    print(b, v)
print("IDS", " ".join(i for v in pick.values() for i, _ in v))
