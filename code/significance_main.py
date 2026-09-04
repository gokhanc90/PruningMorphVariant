"""Significance tests over the main table.

Every comparison is paired PER QUERY (the same queries on both sides).
  - paired t-test
  - permutation (sign) test, 10,000 replicates
  - Holm-Bonferroni correction (within each family of comparisons)

The comparisons that matter:
  A) Pruned vs its own base stemmer   <- the main claim of the paper
  B) Pruned vs NoStem
  C) Pruned vs QBS    (the strongest classical baseline)
  D) Pruned vs MLFN   (the neural baseline a reviewer asked for)
  E) LOO vs fixed     (a comparison of protocols)
"""
import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
OUT = Path(__file__).parent / "out"
GRID = [round(x, 1) for x in np.arange(0, 1.01, 0.1)]
PAIRS = [(b, t) for b in GRID for t in GRID]
TRACKS = {"CW09B": ["WT09", "WT10", "WT11", "WT12"], "NTCIR": ["WWW13", "WWW14"]}
MODELS = ["BM25k1.2b0.75", "DPH"]
METRICS = ["ndcg10", "err10", "map"]
FIXED = (0.5, 0.6)
rng = np.random.default_rng(0)


def per_query(coll, metric):
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


def perm_p(d, n=10000):
    d = np.asarray(d, float)
    d = d[d != 0]
    if len(d) == 0:
        return 1.0
    obs = abs(d.mean())
    signs = rng.choice([-1.0, 1.0], size=(n, len(d)))
    return float((np.abs((signs * d).mean(axis=1)) >= obs).mean())


COMPARISONS = []
for stem in ["Porter", "KStem"]:
    for prot in ["LOO", "fixed"]:
        p = f"Pruned-{stem} ({prot})"
        COMPARISONS += [("A base stemmer", p, stem),
                        ("B NoStem", p, "NoStem"),
                        ("C QBS", p, f"QBS-{stem}"),
                        ("D MLFN", p, f"MLFN-{stem}")]
    COMPARISONS.append(("E protocol", f"Pruned-{stem} (LOO)", f"Pruned-{stem} (fixed)"))

rows = []
for coll in TRACKS:
    for metric in METRICS:
        pq = per_query(coll, metric)
        for model in MODELS:
            for fam, a, b in COMPARISONS:
                if (a, model) not in pq or (b, model) not in pq:
                    continue
                da, db = pq[(a, model)], pq[(b, model)]
                keys = sorted(set(da) & set(db))
                x = np.array([da[k] for k in keys])
                y = np.array([db[k] for k in keys])
                d = x - y
                if len(d) < 10 or np.allclose(d, 0):
                    continue
                rows.append(dict(family=fam, collection=coll, model=model, metric=metric,
                                 A=a, B=b, n=len(d), nz=int((d != 0).sum()),
                                 mean_A=x.mean(), mean_B=y.mean(),
                                 delta=d.mean(), rel=100 * d.mean() / y.mean(),
                                 wins=int((d > 0).sum()), losses=int((d < 0).sum()),
                                 p_t=stats.ttest_rel(x, y).pvalue, p_perm=perm_p(d)))

r = pd.DataFrame(rows)

# Holm-Bonferroni: one family of 12 settings for each (family, A)
r["p_holm"] = np.nan
for (fam, a), g in r.groupby(["family", "A"]):
    p = g.p_perm.to_numpy()
    order = np.argsort(p)
    k, run, adj = len(p), 0.0, np.empty(len(p))
    for i, idx in enumerate(order):
        run = max(run, (k - i) * p[idx])
        adj[idx] = min(run, 1.0)
    r.loc[g.index, "p_holm"] = adj
r.to_csv(OUT / "significance_main.csv", index=False)

print("=" * 104)
print("SIGNIFICANCE -- summary  (12 settings: 2 collections x 2 models x 3 metrics)")
print("=" * 104)
print(f"  {'comparison':46s} {'mean %':>8s} {'positive':>9s} "
      f"{'p<.05 raw':>10s} {'p<.05 Holm':>11s}")
for (fam, a, b), g in r.groupby(["family", "A", "B"]):
    print(f"  {a + '  vs  ' + b:46s} {g.rel.mean():+8.2f} "
          f"{int((g.delta > 0).sum()):5d}/{len(g):<3d} "
          f"{int((g.p_perm < 0.05).sum()):10d} {int((g.p_holm < 0.05).sum()):11d}")

print("\n" + "=" * 104)
print("SETTINGS THAT REMAIN SIGNIFICANT AFTER HOLM")
print("=" * 104)
sig = r[r.p_holm < 0.05].sort_values("p_holm")
if sig.empty:
    print("  none")
else:
    print(f"  {'A':24s} {'B':16s} {'coll':7s} {'model':14s} {'metric':7s} "
          f"{'delta':>9s} {'%':>7s} {'p_holm':>8s}")
    for _, x in sig.iterrows():
        print(f"  {x.A:24s} {x.B:16s} {x.collection:7s} {x.model:14s} {x.metric:7s} "
              f"{x.delta:+9.4f} {x.rel:+6.1f}% {x.p_holm:8.4f}")

print(f"\n  Note: the systems produce different scores on only a subset of the queries "
      f"({r.nz.mean():.0f}/{r.n.mean():.0f} on average), which lowers the power of the tests.")
print(f"\n-> out/significance_main.csv")
