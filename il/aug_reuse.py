"""After re-running the augmentation: keep the CARLA-check rows of copies that are byte-identical to the v1 copies
(scene noise and ego offset are unchanged, same seeds), so only new / changed copies are checked again."""
import json
import os

import pandas as pd

IL = "Scenariosssss/IL"
old = {s["id"]: json.dumps(s, sort_keys=True) for s in json.load(open(f"{IL}/aug_v1/scenarios_aug.json"))}
new = {s["id"]: json.dumps(s, sort_keys=True) for s in json.load(open(f"{IL}/scenarios_aug.json"))}
same = {i for i, t in new.items() if old.get(i) == t}
rep = pd.read_csv(f"{IL}/aug_v1/replay_aug_cureE.csv", dtype={"id": str})
kept = rep[rep.id.isin(same)]
kept.to_csv(f"{IL}/replay_aug_cureE.csv", index=False)
types = pd.Series({s["id"]: s["aug"]["type"] for s in json.load(open(f"{IL}/scenarios_aug.json"))})
print(f"{len(new):,} copies: {len(same):,} identical to v1 (check reused), {len(new) - len(same):,} new or changed to check")
print("to check by type:", types[~types.index.isin(same)].value_counts().to_dict())
