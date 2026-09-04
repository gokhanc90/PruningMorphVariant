"""Batch evaluation of the 11x11 grid.

Only UNIQUE run directories are evaluated; the (beta, Th) -> synonym file
mapping comes from grid_manifest.json. Empty sets (every variant pruned) are
filled in with the NoStem baseline, because with an empty synonym file the
system is exactly equivalent to NoStem.

nDCG@10 / ERR@10 : gdeval_py (verified 492/492 against gdeval.pl)
MAP              : pytrec_eval (verified 197/197 against trec_eval)
"""
import json, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import pytrec_eval

sys.path.insert(0, str(Path(__file__).parent))
from gdeval_py import load_qrels, ideal_dcg, evaluate as gd_eval

warnings.filterwarnings("ignore")
TFD = Path(r"SET_TFD_HOME_HERE")
OUT = Path(__file__).parent / "out"
COLL = sys.argv[1] if len(sys.argv) > 1 else "CW09B"

QRELS = {"WT09": "qrels.web.1-50.txt", "WT10": "qrels.web.51-100.txt",
         "WT11": "qrels.web.101-150.txt", "WT12": "qrels.web.151-200.txt",
         "WWW13": "qrels.www.1-100.txt", "WWW14": "qrels.www.101-180.txt"}
TRACKS = ["WT09", "WT10", "WT11", "WT12"] if COLL == "CW09B" else ["WWW13", "WWW14"]
MODELS = ["BM25k1.2b0.75", "DPH"]
TAG = {"KStem": "SynonymKStem", "SnowballEng": "SynonymSnowballEng"}

manifest = json.loads((OUT / "grid_manifest.json").read_text(encoding="utf-8"))
manifest = [m for m in manifest if m["collection"] == COLL]

# --- qrels once
Q = {}
for tr in TRACKS:
    j = load_qrels(TFD / "topics-and-qrels" / QRELS[tr])
    Q[tr] = (j, ideal_dcg(j, 10),
             pytrec_eval.RelevanceEvaluator(j, {"map"}))


def eval_run(path, track):
    j, idl, ev = Q[track]
    gd = gd_eval(path, j, idl, 10)
    run = {}
    for line in open(path):
        p = line.split()
        if len(p) >= 5:
            run.setdefault(p[0], {})[p[2]] = float(p[4])
    mp = ev.evaluate(run)
    return {q: (n, e, mp.get(q, {}).get("map", np.nan)) for q, (n, e) in gd.items()}


# --- unique run directories
uniq = {}
for m in manifest:
    if m["empty"]:
        continue
    key = (m["stemmer"], m["synonym_file"])
    uniq.setdefault(key, m)
print(f"{COLL}: {len(manifest)} configurations -> {len(uniq)} unique runs")

rows = []
t0 = time.time()
for i, ((stem, synfile), m) in enumerate(uniq.items(), 1):
    runs_dir = TFD / COLL / ("G_" + Path(synfile).stem) / TAG[stem]
    for track in TRACKS:
        for model in MODELS:
            p = runs_dir / track / f"{model}_contents_{TAG[stem]}_OR_all.txt"
            if not p.exists():
                print(f"  EKSIK: {p.relative_to(TFD)}")
                continue
            for q, (n, e, mp) in eval_run(p, track).items():
                rows.append((stem, synfile, track, model, q, n, e, mp))
    if i % 20 == 0:
        el = time.time() - t0
        print(f"  {i}/{len(uniq)}  {el/60:.1f} dk gecti, "
              f"tahmini kalan {el/i*(len(uniq)-i)/60:.1f} dk")

df = pd.DataFrame(rows, columns=["stemmer", "synonym_file", "track", "model",
                                 "qid", "ndcg10", "err10", "map"])

# --- empty configurations: fill them in from the NoStem baseline
empt = [m for m in manifest if m["empty"]]
if empt:
    nos = []
    for track in TRACKS:
        p = TFD / COLL / "B_NoStem" / "NoStem" / track / f"{{}}_contents_NoStem_OR_all.txt"
        for model in MODELS:
            pp = Path(str(p).format(model))
            if pp.exists():
                for q, (n, e, mp) in eval_run(pp, track).items():
                    nos.append((track, model, q, n, e, mp))
    nosdf = pd.DataFrame(nos, columns=["track", "model", "qid", "ndcg10", "err10", "map"])
    seen = set()
    add = []
    for m in empt:
        k = (m["stemmer"], m["synonym_file"])
        if k in seen:
            continue
        seen.add(k)
        t = nosdf.copy()
        t["stemmer"] = m["stemmer"]
        t["synonym_file"] = m["synonym_file"]
        add.append(t)
    if add:
        df = pd.concat([df] + add, ignore_index=True)
    print(f"  empty configurations: {len(empt)} -> {len(seen)} unique, filled in from NoStem")

df.to_parquet(OUT / f"grid_perquery_{COLL}.parquet", index=False)
print(f"\n-> out/grid_perquery_{COLL}.parquet  ({len(df)} rows, "
      f"{(time.time()-t0)/60:.1f} dk)")

agg = df.groupby(["stemmer", "synonym_file", "track", "model"])[
    ["ndcg10", "err10", "map"]].mean().reset_index()
agg.to_csv(OUT / f"grid_agg_{COLL}.csv", index=False)
print(f"-> out/grid_agg_{COLL}.csv  ({len(agg)} rows)")
