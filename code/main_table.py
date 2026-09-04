"""MAIN RESULTS TABLE -- the counterpart of Tables 3/4/5 in the paper.

Systems:
  NoStem, Porter, KStem, HPS            -> baseline_perquery_<coll>.csv
  QBS-Porter, QBS-KStem                 -> weighted_perquery.csv
  MLFN-Porter, MLFN-KStem               -> weighted_perquery.csv
  Pruned-*  (LOO)                       -> grid + the LOO protocol
  Pruned-*  (fixed beta=0.5, Th=0.6)    -> grid, a single configuration

Every number comes from the corrected scoring (sub-word mean pooling) and the
verified evaluation pipeline.
"""
import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
OUT = Path(__file__).parent / "out"
GRID = [round(x, 1) for x in np.arange(0, 1.01, 0.1)]
PAIRS = [(b, t) for b in GRID for t in GRID]
TRACKS = {"CW09B": ["WT09", "WT10", "WT11", "WT12"], "NTCIR": ["WWW13", "WWW14"]}
MODELS = ["BM25k1.2b0.75", "DPH"]
METRICS = ["ndcg10", "err10", "map"]
# Fixed configuration: the point where three independent protocols meet (the LOO
# mean, the leave-one-track-out mode, and the joint LTO+CC mode). It was derived
# WITHOUT looking at the target settings -- scanning all 121 points and keeping
# the best would be an oracle leak.
FIXED = (0.5, 0.6)
STEM_LBL = {"SnowballEng": "Porter", "KStem": "KStem"}


def topk_pick(means, k=5):
    idx = np.argsort(-means, kind="stable")[:k]
    b = np.array([PAIRS[i][0] for i in idx])
    return int(idx[int(np.argmin(np.abs(b - 0.5)))])


def grid_frames(coll):
    df = pd.read_parquet(OUT / f"grid_perquery_{coll}.parquet")
    man = [m for m in json.loads((OUT / "grid_manifest.json").read_text(encoding="utf-8"))
           if m["collection"] == coll]
    key = pd.DataFrame([{"stemmer": m["stemmer"], "synonym_file": m["synonym_file"],
                         "beta": m["beta"], "th": m["th"]} for m in man])
    return df.merge(key, on=["stemmer", "synonym_file"], how="inner")


rows = []
for coll in TRACKS:
    # --- baselines
    b = pd.read_csv(OUT / f"baseline_perquery_{coll}.csv")
    for sysname in ["NoStem", "Porter", "KStem", "HPS"]:
        d = b[b.system == sysname]
        for (track, model), g in d.groupby(["track", "model"]):
            for m in METRICS:
                rows.append(dict(collection=coll, system=sysname, track=track,
                                 model=model, metric=m, score=g[m].mean()))

    # --- weighted systems
    w = pd.read_csv(OUT / "weighted_perquery.csv")
    w = w[w.collection == coll]
    for (sysname, track, model), g in w.groupby(["system", "track", "model"]):
        lbl = sysname.replace("-SnowballEng", "-Porter").replace("MLFN-ART", "MLFN")
        for m in METRICS:
            rows.append(dict(collection=coll, system=lbl, track=track,
                             model=model, metric=m, score=g[m].mean()))

    # --- pruned: LOO and fixed
    gf = grid_frames(coll)
    for stem in ["SnowballEng", "KStem"]:
        for model in MODELS:
            for metric in METRICS:
                for track in TRACKS[coll]:
                    p = gf[(gf.stemmer == stem) & (gf.model == model) &
                           (gf.track == track)].pivot_table(
                        index=["beta", "th"], columns="qid", values=metric)
                    p = p.reindex(pd.MultiIndex.from_tuples(PAIRS, names=["beta", "th"]))
                    G = p.to_numpy(float)
                    # LOO
                    n = G.shape[1]
                    tot = np.nansum(G, axis=1)
                    res = [G[topk_pick((tot - G[:, q]) / (n - 1)), q] for q in range(n)]
                    rows.append(dict(collection=coll, system=f"Pruned-{STEM_LBL[stem]} (LOO)",
                                     track=track, model=model, metric=metric,
                                     score=float(np.nanmean(res))))
                    # fixed
                    i = PAIRS.index(FIXED)
                    rows.append(dict(collection=coll,
                                     system=f"Pruned-{STEM_LBL[stem]} (fixed)",
                                     track=track, model=model, metric=metric,
                                     score=float(np.nanmean(G[i]))))

df = pd.DataFrame(rows)
df.to_csv(OUT / "main_table.csv", index=False)

SYS = ["NoStem", "Porter", "KStem", "HPS", "QBS-Porter", "QBS-KStem",
       "MLFN-Porter", "MLFN-KStem",
       "Pruned-Porter (LOO)", "Pruned-KStem (LOO)",
       "Pruned-Porter (fixed)", "Pruned-KStem (fixed)"]

for coll in TRACKS:
    for metric in METRICS:
        print("\n" + "=" * 104)
        print(f"{coll}  --  {metric.upper()}   (fixed = beta {FIXED[0]}, Th {FIXED[1]})")
        print("=" * 104)
        for model in MODELS:
            d = df[(df.collection == coll) & (df.metric == metric) & (df.model == model)]
            p = d.pivot_table(index="system", columns="track", values="score")
            p = p.reindex([s for s in SYS if s in p.index])
            p["MEAN"] = p.mean(axis=1)
            print(f"\n  [{model}]")
            print("    " + "system".ljust(24) + "".join(f"{t:>9s}" for t in TRACKS[coll])
                  + f"{'MEAN':>10s}")
            for s, r in p.iterrows():
                print(f"    {s:24s}" + "".join(f"{r[t]:9.4f}" for t in TRACKS[coll])
                      + f"{r['MEAN']:10.4f}")

# ---------------- mean rank ----------------
print("\n" + "=" * 104)
print("GLOBAL MEAN RANK  (2 collections x 2 models x 3 metrics = 12 settings, over track means)")
print("=" * 104)
ranks = {}
for coll in TRACKS:
    for model in MODELS:
        for metric in METRICS:
            d = df[(df.collection == coll) & (df.model == model) & (df.metric == metric)]
            s = d.groupby("system").score.mean().sort_values(ascending=False)
            srt = list(s.values)
            for sysname, v in s.items():
                ranks.setdefault(sysname, []).append(srt.index(v) + 1)
res = sorted(((np.mean(v), s, v) for s, v in ranks.items()))
print(f"  {'system':24s} {'mean rank':>10s} {'times best':>11s}")
for avg, s, v in res:
    print(f"  {s:24s} {avg:10.2f} {sum(1 for x in v if x == 1):11d}")
print(f"\n-> out/main_table.csv")
