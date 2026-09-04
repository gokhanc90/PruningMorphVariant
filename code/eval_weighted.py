"""Evaluates the weighted systems (QBS, MLFN-ART).

Both were run with the same driver, the same tf scheme and the same ART
formula; the ONLY difference is where NPMI comes from (computed from the index
versus predicted by MLFN).
"""
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import pytrec_eval

sys.path.insert(0, str(Path(__file__).parent))
from gdeval_py import load_qrels, ideal_dcg, evaluate as gd_eval

warnings.filterwarnings("ignore")
TFD = Path(r"SET_TFD_HOME_HERE")
OUT = Path(__file__).parent / "out"
QRELS = {"WT09": "qrels.web.1-50.txt", "WT10": "qrels.web.51-100.txt",
         "WT11": "qrels.web.101-150.txt", "WT12": "qrels.web.151-200.txt",
         "WWW13": "qrels.www.1-100.txt", "WWW14": "qrels.www.101-180.txt"}
TRACKS = {"CW09B": ["WT09", "WT10", "WT11", "WT12"], "NTCIR": ["WWW13", "WWW14"]}
MODELS = ["BM25k1.2b0.75", "DPH"]
TAG = {"SnowballEng": "SynonymSnowballEngQBS", "KStem": "SynonymKStemQBS"}

Q = {}
def qq(tr):
    if tr not in Q:
        j = load_qrels(TFD / "topics-and-qrels" / QRELS[tr])
        Q[tr] = (j, ideal_dcg(j, 10), pytrec_eval.RelevanceEvaluator(j, {"map"}))
    return Q[tr]


def ev(path, track):
    j, idl, e = qq(track)
    gd = gd_eval(path, j, idl, 10)
    run = {}
    for line in open(path):
        p = line.split()
        if len(p) >= 5:
            run.setdefault(p[0], {})[p[2]] = float(p[4])
    mp = e.evaluate(run)
    return {q: (n, er, mp.get(q, {}).get("map", np.nan)) for q, (n, er) in gd.items()}


rows = []
for coll in TRACKS:
    for system, prefix in [("QBS", "W_QBS_"), ("MLFN-ART", "W_MLFN_")]:
        for stem in ["SnowballEng", "KStem"]:
            base = TFD / coll / (prefix + stem) / TAG[stem]
            if not base.exists():
                print(f"  not found: {base.relative_to(TFD)}")
                continue
            for track in TRACKS[coll]:
                for model in MODELS:
                    p = base / track / f"{model}_contents_{TAG[stem]}_OR_all.txt"
                    if not p.exists():
                        print(f"  missing: {p.relative_to(TFD)}")
                        continue
                    for q, (n, e, m) in ev(p, track).items():
                        rows.append(dict(collection=coll, system=f"{system}-{stem}",
                                         base_system=system, stemmer=stem,
                                         track=track, model=model, qid=q,
                                         ndcg10=n, err10=e, map=m))
            print(f"  evaluated: {coll}/{system}/{stem}")

df = pd.DataFrame(rows)
df.to_csv(OUT / "weighted_perquery.csv", index=False)
agg = df.groupby(["collection", "system", "track", "model"])[
    ["ndcg10", "err10", "map"]].mean().reset_index()
agg.to_csv(OUT / "weighted_agg.csv", index=False)

print("\n" + "=" * 92)
print("QBS (full NPMI) vs MLFN-ART (predicted NPMI)  --  same formula, same tf scheme")
print("=" * 92)
for coll in TRACKS:
    for metric in ["ndcg10", "err10", "map"]:
        for model in MODELS:
            d = agg[(agg.collection == coll) & (agg.model == model)]
            if d.empty:
                continue
            piv = d.pivot_table(index="system", columns="track", values=metric)
            piv["MEAN"] = piv.mean(axis=1)
            print(f"\n  {coll} / {model} / {metric}")
            for s, r in piv.sort_values("MEAN", ascending=False).iterrows():
                print(f"    {s:22s}" + "".join(f"{r[t]:9.4f}" for t in TRACKS[coll])
                      + f"{r['MEAN']:10.4f}")

print(f"\n-> out/weighted_agg.csv")
