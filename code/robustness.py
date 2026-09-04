"""Risk-sensitive evaluation (TRisk) + significance tests.

TRisk (Dincer et al.): a line-by-line port of TRisk.m.
    diff = run - base;  diff[diff<0] *= (alpha+1)
    TRisk = mean(diff) / sqrt(var(diff)/n)
Baseline: NoStem (as in the paper).
"""
import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
OUT = Path(__file__).parent / "out"
TAB = Path(__file__).parent / "tablolar"
TAB.mkdir(exist_ok=True)
GRID = [round(x, 1) for x in np.arange(0, 1.01, 0.1)]
PAIRS = [(b, t) for b in GRID for t in GRID]
TRACKS = {"CW09B": ["WT09", "WT10", "WT11", "WT12"], "NTCIR": ["WWW13", "WWW14"]}
MODELS = ["BM25k1.2b0.75", "DPH"]
FIXED = (0.5, 0.6)      # see the paper, Section 5.3 -- derived without looking at the target
ALPHAS = [0, 1, 2, 3, 4, 5]


def trisk(base, run, alpha):
    """Port of TRisk.m. base/run: score arrays over the same queries."""
    d = np.asarray(run, float) - np.asarray(base, float)
    d = np.where(d < 0, (alpha + 1) * d, d)
    n = len(d)
    m = d.mean()
    s1 = np.sum((d - m) ** 2)
    s2 = np.sum(d - m)
    var = (s1 - (s2 * s2 / n)) / (n - 1)
    return float(m / np.sqrt(var / n)) if var > 0 else np.nan


def per_query(coll, metric="ndcg10"):
    """-> {system: {(track, qid): score}}"""
    out = {}
    b = pd.read_csv(OUT / f"baseline_perquery_{coll}.csv")
    for s in ["NoStem", "Porter", "KStem", "HPS"]:
        d = b[b.system == s]
        for model in MODELS:
            dd = d[d.model == model]
            out[(s, model)] = dict(zip(zip(dd.track, dd.qid.astype(str)), dd[metric]))
    w = pd.read_csv(OUT / "weighted_perquery.csv")
    w = w[w.collection == coll]
    for s, g in w.groupby("system"):
        lbl = s.replace("-SnowballEng", "-Porter").replace("MLFN-ART", "MLFN")
        for model in MODELS:
            dd = g[g.model == model]
            out[(lbl, model)] = dict(zip(zip(dd.track, dd.qid.astype(str)), dd[metric]))

    # pruned: LOO and fixed
    gp = pd.read_parquet(OUT / f"grid_perquery_{coll}.parquet")
    man = [m for m in json.loads((OUT / "grid_manifest.json").read_text(encoding="utf-8"))
           if m["collection"] == coll]
    key = pd.DataFrame([{"stemmer": m["stemmer"], "synonym_file": m["synonym_file"],
                         "beta": m["beta"], "th": m["th"]} for m in man])
    gp = gp.merge(key, on=["stemmer", "synonym_file"], how="inner")
    for stem, lbl in [("SnowballEng", "Porter"), ("KStem", "KStem")]:
        for model in MODELS:
            loo_d, fix_d = {}, {}
            for track in TRACKS[coll]:
                p = gp[(gp.stemmer == stem) & (gp.model == model) &
                       (gp.track == track)].pivot_table(
                    index=["beta", "th"], columns="qid", values=metric)
                p = p.reindex(pd.MultiIndex.from_tuples(PAIRS, names=["beta", "th"]))
                G, qids = p.to_numpy(float), list(p.columns)
                n = G.shape[1]
                tot = np.nansum(G, axis=1)
                for j, q in enumerate(qids):
                    m_ = (tot - G[:, j]) / (n - 1)
                    idx = np.argsort(-m_, kind="stable")[:5]
                    bb = np.array([PAIRS[i][0] for i in idx])
                    i = int(idx[int(np.argmin(np.abs(bb - 0.5)))])
                    loo_d[(track, str(q))] = G[i, j]
                    fix_d[(track, str(q))] = G[PAIRS.index(FIXED), j]
            out[(f"Pruned-{lbl} (LOO)", model)] = loo_d
            out[(f"Pruned-{lbl} (fixed)", model)] = fix_d
    return out


SYS = ["Porter", "KStem", "HPS", "QBS-Porter", "QBS-KStem", "MLFN-Porter", "MLFN-KStem",
       "Pruned-Porter (LOO)", "Pruned-KStem (LOO)",
       "Pruned-Porter (fixed)", "Pruned-KStem (fixed)"]

rows, trows = [], []
for coll in TRACKS:
    pq = per_query(coll)
    for model in MODELS:
        base = pq[("NoStem", model)]
        keys = sorted(base)
        for s in SYS:
            if (s, model) not in pq:
                continue
            d = pq[(s, model)]

            # --- whole collection
            common = [k for k in keys if k in d]
            bb = [base[k] for k in common]
            rr = [d[k] for k in common]
            row = dict(collection=coll, model=model, system=s, n=len(common))
            for a in ALPHAS:
                row[f"a{a}"] = trisk(bb, rr, a)
            row["p_ttest"] = stats.ttest_rel(rr, bb).pvalue
            row["delta"] = np.mean(rr) - np.mean(bb)
            rows.append(row)

            # --- PER TRACK
            for track in TRACKS[coll]:
                ck = [k for k in keys if k[0] == track and k in d]
                if len(ck) < 5:
                    continue
                bt = [base[k] for k in ck]
                rt = [d[k] for k in ck]
                tr_ = dict(collection=coll, track=track, model=model,
                           system=s, n=len(ck))
                for a in ALPHAS:
                    tr_[f"a{a}"] = trisk(bt, rt, a)
                tr_["p_ttest"] = stats.ttest_rel(rt, bt).pvalue
                tr_["delta"] = np.mean(rt) - np.mean(bt)
                trows.append(tr_)

r = pd.DataFrame(rows)
r.to_csv(OUT / "trisk.csv", index=False)
tr = pd.DataFrame(trows)
tr.to_csv(OUT / "trisk_track.csv", index=False)

print("=" * 108)
print("TRisk -- PER TRACK, against NoStem, nDCG@10")
print("=" * 108)
TRACK_LBL = {"WT09": "WT09", "WT10": "WT10", "WT11": "WT11", "WT12": "WT12",
             "WWW13": "NTCIR-13", "WWW14": "NTCIR-14"}
for coll in TRACKS:
    for track in TRACKS[coll]:
        for model in MODELS:
            d = tr[(tr.track == track) & (tr.model == model)]
            if d.empty:
                continue
            print(f"\n  {TRACK_LBL[track]} / {model}   (n={int(d.n.iloc[0])})")
            print(f"    {'system':24s}" + "".join(f"{'a=' + str(a):>8s}" for a in ALPHAS)
                  + f"{'delta':>9s}")
            for _, x in d.sort_values("a5", ascending=False).iterrows():
                print(f"    {x.system:24s}"
                      + "".join(f"{x['a' + str(a)]:8.2f}" for a in ALPHAS)
                      + f"{x.delta:+9.4f}")

print("=" * 104)
print("TRisk  --  against NoStem, nDCG@10   (the larger alpha, the heavier the risk penalty)")
print("=" * 104)
for coll in TRACKS:
    for model in MODELS:
        d = r[(r.collection == coll) & (r.model == model)]
        if d.empty:
            continue
        print(f"\n  {coll} / {model}")
        print(f"    {'system':24s}" + "".join(f"{'a=' + str(a):>8s}" for a in ALPHAS)
              + f"{'delta':>9s}{'p':>8s}")
        for _, x in d.sort_values("a5", ascending=False).iterrows():
            star = "*" if x.p_ttest < 0.05 else " "
            print(f"    {x.system:24s}" + "".join(f"{x['a' + str(a)]:8.2f}" for a in ALPHAS)
                  + f"{x.delta:+9.4f}{x.p_ttest:8.3f}{star}")

print("\n" + "=" * 104)
print("SUMMARY: systems that stay positive at alpha=5 (the strictest risk penalty)")
print("=" * 104)
pos = r.groupby("system").a5.agg(["mean", lambda s: (s > 0).sum(), "count"])
pos.columns = ["a5_mean", "positive", "total"]
for s, x in pos.sort_values("a5_mean", ascending=False).iterrows():
    print(f"  {s:24s} a5 mean={x.a5_mean:7.2f}   positive: {int(x.positive)}/{int(x.total)}")
print(f"\n-> out/trisk.csv")
