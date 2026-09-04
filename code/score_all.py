"""Scores every pair with the six representations, for both stemmers, on both
collections.
"""
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from scorers import BertScorer, SbertScorer, cosine_from_vectors

warnings.filterwarnings("ignore")
PMV = Path(__file__).parent.parent / "npmi"
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

# (collection, stemmer) -> npmi input file
ALL_JOBS = [("CW09B", "SnowballEng", "npmi_CW09B_SnowballEng.csv"),
            ("CW09B", "KStem",       "npmi_CW09B_KStem.csv"),
            ("NTCIR", "SnowballEng", "npmi_NTCIR_SnowballEng.csv"),
            ("NTCIR", "KStem",       "npmi_NTCIR_KStem.csv")]

# a subset can be selected by argument -> several processes can run in parallel
if len(sys.argv) > 1:
    sel = [int(x) for x in sys.argv[1].split(",")]
    JOBS = [ALL_JOBS[i] for i in sel]
    torch.set_num_threads(4)
else:
    JOBS = ALL_JOBS
print(f"jobs to process: {[(c, s) for c, s, _ in JOBS]}")

print("loading models...")
bert = BertScorer()
sbert = SbertScorer()
z = np.load(OUT / "fasttext_vectors.npz", allow_pickle=True)
FT = {str(w): v for w, v in zip(z["words"], z["vectors"])}
print("ready.\n")



for coll, stem, fname in JOBS:
    t0 = time.time()
    df = pd.read_csv(PMV / fname)
    df = df.drop_duplicates(subset=["Term", "Morph"], keep="last").reset_index(drop=True)
    df["Term"] = df["Term"].astype(str)
    df["Morph"] = df["Morph"].astype(str)
    pairs = list(zip(df.Term, df.Morph))
    words = pd.unique(pd.concat([df.Term, df.Morph]))
    topics = pd.read_csv(OUT / f"topics_{coll}.tsv", sep="\t", header=0)
    qmap = dict(zip(topics.qid.astype(int), topics["query"].astype(str)))
    print(f"--- {coll}/{stem}: {len(df)} pairs, {len(words)} words")

    res = {}
    v_cls, v_sub = bert.word_vectors(words)
    res["cls_masked"] = cosine_from_vectors(v_cls, pairs)
    res["subword_mean"] = cosine_from_vectors(v_sub, pairs)
    res["bert_static"] = cosine_from_vectors(bert.static_vectors(words), pairs)
    res["sbert"] = cosine_from_vectors(sbert.word_vectors(words), pairs)
    res["fasttext"] = cosine_from_vectors({w: FT[w] for w in words if w in FT}, pairs)

    items = [(qmap.get(int(qid), ""), t, m)
             for qid, t, m in zip(df.QID, df.Term, df.Morph)]
    res["query_ctx"] = bert.query_context_sims(items)

    o = pd.DataFrame({"QID": df.QID, "Term": df.Term, "Morph": df.Morph,
                      "npmi": df["npmi"], **res})
    o.to_csv(OUT / f"scores_{coll}_{stem}.csv", index=False)
    cov = {k: f"{100*np.isfinite(v).mean():.1f}%" for k, v in res.items()}
    print(f"    coverage: {cov}")
    print(f"    -> scores_{coll}_{stem}.csv   ({time.time()-t0:.0f} s)\n")
print("DONE")
