"""Scores every pair with the six representations, for both stemmers, on both
collections.

cls_orig is pair-dependent (the padding length depends on the variant it is
matched with), so pairs are GROUPED by their own maximum length and processed
in batches. This preserves the original semantics exactly while running about
50x faster.
"""
import sys, time, warnings
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from scorers import BertScorer, SbertScorer, cosine_from_vectors

warnings.filterwarnings("ignore")
PMV = Path(r"SET_NPMI_WORKBOOK_DIR_HERE")
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

# (collection, stemmer) -> (npmi workbook, sheet)
ALL_JOBS = [("CW09B", "SnowballEng", "npmi-CW09B.xlsx", "SnowballEng"),
            ("CW09B", "KStem",       "npmi-CW09B.xlsx", "KStem"),
            ("NTCIR", "SnowballEng", "npmi-NTCIR-Bert.xlsx", "SnowballEng"),
            ("NTCIR", "KStem",       "npmi-NTCIR-Bert.xlsx", "KStem")]

# a subset can be selected by argument -> several processes can run in parallel
if len(sys.argv) > 1:
    sel = [int(x) for x in sys.argv[1].split(",")]
    JOBS = [ALL_JOBS[i] for i in sel]
    torch.set_num_threads(4)
else:
    JOBS = ALL_JOBS
print(f"jobs to process: {[(c, s) for c, s, _, _ in JOBS]}")

print("loading models...")
bert = BertScorer()
sbert = SbertScorer()
z = np.load(OUT / "fasttext_vectors.npz", allow_pickle=True)
FT = {str(w): v for w, v in zip(z["words"], z["vectors"])}
print("ready.\n")


def cls_orig_batched(pairs, bs=128):
    """Original semantics: every pair is zero-padded to ITS OWN maximum length,
    with NO mask. Pairs of equal length are batched together for speed."""
    tok = bert.tok
    lens = {}
    for w1, w2 in set(pairs):
        l1 = len(tok.tokenize(str(w1))) + 2
        l2 = len(tok.tokenize(str(w2))) + 2
        lens[(w1, w2)] = max(min(l1, 16), min(l2, 16))
    groups = defaultdict(list)
    for i, p in enumerate(pairs):
        groups[lens[p]].append(i)

    out = np.full(len(pairs), np.nan)
    for L, idxs in groups.items():
        for s in range(0, len(idxs), bs):
            chunk = idxs[s:s + bs]
            rows = []
            for i in chunk:
                for w in pairs[i]:
                    e = tok(str(w), add_special_tokens=True, max_length=16,
                            truncation=True, return_tensors="pt")["input_ids"]
                    rows.append(torch.nn.ZeroPad2d((0, L - e.shape[1], 0, 0))(e)
                                if e.shape[1] < L else e[:, :L])
            t = torch.cat(rows, 0)
            with torch.no_grad():
                v = bert.model(t).last_hidden_state[:, 0, :].numpy()
            v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
            for k, i in enumerate(chunk):
                out[i] = float(v[2 * k] @ v[2 * k + 1])
    return out


for coll, stem, fname, sheet in JOBS:
    t0 = time.time()
    df = pd.read_excel(PMV / fname, sheet_name=sheet, header=0)
    df = df.drop_duplicates(subset=["Term", "Morph"], keep="last").reset_index(drop=True)
    df["Term"] = df["Term"].astype(str)
    df["Morph"] = df["Morph"].astype(str)
    pairs = list(zip(df.Term, df.Morph))
    words = pd.unique(pd.concat([df.Term, df.Morph]))
    topics = pd.read_csv(OUT / f"topics_{coll}.tsv", sep="\t", header=0)
    qmap = dict(zip(topics.qid.astype(int), topics["query"].astype(str)))
    print(f"--- {coll}/{stem}: {len(df)} pairs, {len(words)} words")

    res = {}
    res["cls_orig"] = cls_orig_batched(pairs)
    if "bert-base-uncased" in df.columns:            # verify against stored values
        old = df["bert-base-uncased"].to_numpy(float)
        m = np.isfinite(old) & np.isfinite(res["cls_orig"])
        d = np.abs(res["cls_orig"][m] - old[m])
        print(f"    cls_orig vs stored: identical={int((d<1e-4).sum())}/{int(m.sum())} "
              f"max|diff|={d.max():.6f}")

    v_cls, v_sub = bert.word_vectors(words)
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
