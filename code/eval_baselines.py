"""Evaluates the re-run baselines and compares them against Table 3 of the paper.

The baselines do not depend on the BERT scoring at all, so a re-run has to
reproduce the published numbers exactly. When it does, the whole pipeline --
including QBS and HPS -- is verified end to end.
"""
import subprocess, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import pytrec_eval

warnings.filterwarnings("ignore")
TFD = Path(r"SET_TFD_HOME_HERE")
OUT = Path(__file__).parent / "out"
COLL = sys.argv[1] if len(sys.argv) > 1 else "CW09B"

QRELS = {"WT09": "qrels.web.1-50.txt", "WT10": "qrels.web.51-100.txt",
         "WT11": "qrels.web.101-150.txt", "WT12": "qrels.web.151-200.txt",
         "WWW13": "qrels.www.1-100.txt", "WWW14": "qrels.www.101-180.txt"}
TRACKS = ["WT09", "WT10", "WT11", "WT12"] if COLL == "CW09B" else ["WWW13", "WWW14"]
MODELS = ["BM25k1.2b0.75", "DPH"]

# name -> (runs directory, Tag)
BASE = {"NoStem": ("B_NoStem", "NoStem"),
        "Porter": ("B_Porter", "SynonymSnowballEng"),
        "KStem": ("B_KStem", "SynonymKStem"),
        "HPS": ("B_HPS", "SynonymHPS"),
        "QBS-Porter": ("B_QBS-Porter", "SynonymSnowballEngQBS"),
        "QBS-KStem": ("B_QBS-KStem", "SynonymKStemQBS")}

# paper Table 3, CW09B, BM25 nDCG@10
PAPER = {
 ("WT09", "NoStem"): .3119, ("WT09", "Porter"): .2955, ("WT09", "KStem"): .2952,
 ("WT09", "HPS"): .2467, ("WT09", "QBS-Porter"): .3012, ("WT09", "QBS-KStem"): .2995,
 ("WT10", "NoStem"): .1267, ("WT10", "Porter"): .1187, ("WT10", "KStem"): .1181,
 ("WT10", "HPS"): .1122, ("WT10", "QBS-Porter"): .1239, ("WT10", "QBS-KStem"): .1224,
 ("WT11", "NoStem"): .1479, ("WT11", "Porter"): .1286, ("WT11", "KStem"): .1394,
 ("WT11", "HPS"): .1146, ("WT11", "QBS-Porter"): .1363, ("WT11", "QBS-KStem"): .1385,
 ("WT12", "NoStem"): .0529, ("WT12", "Porter"): .0478, ("WT12", "KStem"): .0613,
 ("WT12", "HPS"): .0468, ("WT12", "QBS-Porter"): .0431, ("WT12", "QBS-KStem"): .0524}

_q = {}
def qrels(t):
    if t not in _q:
        d = {}
        for line in open(TFD / "topics-and-qrels" / QRELS[t]):
            p = line.split()
            if len(p) >= 4:
                d.setdefault(p[0], {})[p[2]] = int(p[3])
        _q[t] = d
    return _q[t]


def ev(run, track):
    qp = TFD / "topics-and-qrels" / QRELS[track]
    o = subprocess.run(["perl", str(TFD / "scripts/gdeval.pl"), "-k", "10",
                        str(qp), str(run)], capture_output=True, text=True)
    gd = {}
    for line in o.stdout.splitlines():
        p = line.strip().split(",")
        if len(p) == 4 and p[1] not in ("topic", "amean"):
            gd[p[1]] = (float(p[2]), float(p[3]))
    r = {}
    for line in open(run):
        p = line.split()
        if len(p) >= 5:
            r.setdefault(p[0], {})[p[2]] = float(p[4])
    mp = pytrec_eval.RelevanceEvaluator(qrels(track), {"map"}).evaluate(r)
    return {q: (gd[q][0], gd[q][1], mp.get(q, {}).get("map", np.nan)) for q in gd}


rows = []
for name, (rd, tag) in BASE.items():
    for track in TRACKS:
        for model in MODELS:
            p = TFD / COLL / rd / tag / track / f"{model}_contents_{tag}_OR_all.txt"
            if not p.exists():
                print(f"  missing: {name}/{track}/{model}")
                continue
            for q, (n, e, m) in ev(p, track).items():
                rows.append(dict(system=name, track=track, model=model, qid=q,
                                 ndcg10=n, err10=e, map=m))
    print(f"  evaluated: {name}")

df = pd.DataFrame(rows)
df.to_csv(OUT / f"baseline_perquery_{COLL}.csv", index=False)
agg = df.groupby(["system", "track", "model"])[["ndcg10", "err10", "map"]].mean().reset_index()
agg.to_csv(OUT / f"baseline_agg_{COLL}.csv", index=False)

if COLL == "CW09B":
    print("\n" + "=" * 78)
    print("COMPARISON AGAINST TABLE 3 OF THE PAPER  (BM25, nDCG@10)")
    print("=" * 78)
    print(f"  {'track':7s} {'system':13s} {'re-run':>9s} {'paper':>9s} {'diff':>9s}")
    ok = bad = 0
    for (track, sysname), exp in sorted(PAPER.items()):
        d = agg[(agg.system == sysname) & (agg.track == track) &
                (agg.model == "BM25k1.2b0.75")]
        if d.empty:
            continue
        got = d.ndcg10.iloc[0]
        diff = got - exp
        flag = ""
        if abs(diff) > 5e-5:
            bad += 1; flag = "  <--"
        else:
            ok += 1
        print(f"  {track:7s} {sysname:13s} {got:9.4f} {exp:9.4f} {diff:+9.4f}{flag}")
    print(f"\n  Exact matches: {ok}/{ok+bad}")

print(f"\n-> out/baseline_agg_{COLL}.csv")
