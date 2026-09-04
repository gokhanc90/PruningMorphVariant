"""Generates the synonym files for the 11x11 (beta, Th) grid over subword_mean.

Files with identical content are DE-DUPLICATED: at high thresholds many
configurations produce the same (often empty) set. Only unique files are run,
and results are mapped back through the manifest.
"""
import hashlib, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd

from make_synonyms import prune, write_synonyms, BASE_TAG

warnings.filterwarnings("ignore")
OUT = Path(__file__).parent / "out"
TFD = Path(r"SET_TFD_HOME_HERE")
METHOD = "subword_mean"
GRID = [round(x, 1) for x in np.arange(0, 1.01, 0.1)]
COMBOS = [("CW09B", "KStem"), ("CW09B", "SnowballEng"),
          ("NTCIR", "KStem"), ("NTCIR", "SnowballEng")]

manifest, total_unique, total_all = [], 0, 0
for coll, stem in COMBOS:
    csv = OUT / f"scores_{coll}_{stem}.csv"
    seen = {}          # content hash -> file name
    n_uni = 0
    for b in GRID:
        for t in GRID:
            keep, terms, lb = prune(csv, METHOD, b, t)
            # produce the content (into a temporary file first, for hashing)
            tmp = OUT / "tmp_syn" / "probe.txt"
            tmp.parent.mkdir(exist_ok=True)
            nlines = write_synonyms(coll, stem, terms, tmp)
            content = tmp.read_bytes()
            h = hashlib.md5(content).hexdigest()

            if h in seen:
                fname = seen[h]
                new = False
            else:
                fname = f"{BASE_TAG[stem]}_G_b{b:.1f}_t{t:.1f}.txt"
                (TFD / coll / fname).write_bytes(content)
                seen[h] = fname
                n_uni += 1
                new = True

            manifest.append(dict(collection=coll, stemmer=stem, method=METHOD,
                                 beta=b, th=t, pairs=int(keep.sum()),
                                 lines=nlines, synonym_file=fname,
                                 unique=new, empty=(nlines == 0)))
            total_all += 1
    total_unique += n_uni
    n_empty = sum(1 for m in manifest
                  if m["collection"] == coll and m["stemmer"] == stem and m["empty"])
    print(f"{coll}/{stem:12s} 121 configurations -> {n_uni:3d} unique files "
          f"({n_empty} empty sets)")

(OUT / "grid_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f"\nTOTAL {total_all} configurations -> {total_unique} unique runs")
print(f"saved: {100*(1-total_unique/total_all):.0f}%")
print(f"-> out/grid_manifest.json")
