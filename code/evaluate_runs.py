"""Evaluates the runs under X_<method>_<stemmer> (gdeval.pl + pytrec_eval) and
produces the representation comparison table.

nDCG@10 and ERR@10 come from gdeval.pl; MAP from pytrec_eval over the top 1000.
"""
import subprocess, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import pytrec_eval

warnings.filterwarnings("ignore")
TFD = Path(r"SET_TFD_HOME_HERE")
OUT = Path(__file__).parent / "out"

QRELS = {"WT09": "qrels.web.1-50.txt", "WT10": "qrels.web.51-100.txt",
         "WT11": "qrels.web.101-150.txt", "WT12": "qrels.web.151-200.txt",
         "WWW13": "qrels.www.1-100.txt", "WWW14": "qrels.www.101-180.txt"}
TAG = {"KStem": "SynonymKStem", "SnowballEng": "SynonymSnowballEng"}
METHODS = ["cls_masked", "subword_mean", "bert_static", "sbert", "fasttext", "query_ctx"]
MODELS = ["BM25k1.2b0.75", "DPH"]
COLL = sys.argv[1] if len(sys.argv) > 1 else "CW09B"
TRACKS = ["WT09", "WT10", "WT11", "WT12"] if COLL == "CW09B" else ["WWW13", "WWW14"]

_q = {}
def qrels(track):
    if track not in _q:
        p = TFD / "topics-and-qrels" / QRELS[track]
        d = {}
        for line in open(p):
            pp = line.split()
            if len(pp) >= 4:
                d.setdefault(pp[0], {})[pp[2]] = int(pp[3])
        _q[track] = d
    return _q[track]


def evaluate(run_path, track):
    """-> {qid: (ndcg10, err10, map)}"""
    qp = TFD / "topics-and-qrels" / QRELS[track]
    out = subprocess.run(["perl", str(TFD / "scripts/gdeval.pl"), "-k", "10",
                          str(qp), str(run_path)], capture_output=True, text=True)
    gd = {}
    for line in out.stdout.splitlines():
        pp = line.strip().split(",")
        if len(pp) == 4 and pp[1] not in ("topic", "amean"):
            gd[pp[1]] = (float(pp[2]), float(pp[3]))
    run = {}
    for line in open(run_path):
        pp = line.split()
        if len(pp) >= 5:
            run.setdefault(pp[0], {})[pp[2]] = float(pp[4])
    mp = pytrec_eval.RelevanceEvaluator(qrels(track), {"map"}).evaluate(run)
    return {q: (gd[q][0], gd[q][1], mp.get(q, {}).get("map", np.nan)) for q in gd}


rows = []
for stem in ["SnowballEng", "KStem"]:
    for meth in METHODS:
        base = TFD / COLL / f"X_{meth}_{stem}" / TAG[stem]
        if not base.exists():
            print(f"missing: {base}")
            continue
        for track in TRACKS:
            for model in MODELS:
                rp = base / track / f"{model}_contents_{TAG[stem]}_OR_all.txt"
                if not rp.exists():
                    print(f"  missing: {rp.relative_to(TFD)}")
                    continue
                r = evaluate(rp, track)
                for q, (n, e, m) in r.items():
                    rows.append(dict(collection=COLL, stemmer=stem, method=meth,
                                     track=track, model=model, qid=q,
                                     ndcg10=n, err10=e, map=m))
        print(f"  evaluated: {stem}/{meth}")

df = pd.DataFrame(rows)
df.to_csv(OUT / f"eval_perquery_{COLL}.csv", index=False)
print(f"\n-> out/eval_perquery_{COLL}.csv  ({len(df)} rows)")

agg = df.groupby(["stemmer", "method", "track", "model"])[
    ["ndcg10", "err10", "map"]].mean().reset_index()
agg.to_csv(OUT / f"eval_agg_{COLL}.csv", index=False)

for metric in ["ndcg10", "err10", "map"]:
    print("\n" + "=" * 100)
    print(f"{metric.upper()}  --  {COLL}")
    print("=" * 100)
    for model in MODELS:
        print(f"\n  [{model}]")
        p = agg[agg.model == model].pivot_table(index=["stemmer", "method"],
                                                columns="track", values=metric)
        p["MEAN"] = p.mean(axis=1)
        for stem in ["SnowballEng", "KStem"]:
            sub = p.loc[stem].sort_values("MEAN", ascending=False)
            print(f"    {stem}:")
            hdr = "      " + f"{'method':14s}" + "".join(f"{t:>9s}" for t in TRACKS) + f"{'MEAN':>10s}"
            print(hdr)
            for m, r in sub.iterrows():
                print(f"      {m:14s}" + "".join(f"{r[t]:9.4f}" for t in TRACKS)
                      + f"{r['MEAN']:10.4f}")
