"""Builds the pruned synonym files and verifies them by diffing against the
existing ones.

Pipeline:
  LOWERBOUND = Q1 - 1.5*(Q3-Q1)          (on CS, the interquartile rule)
  pairs with CS > LOWERBOUND survive      (first stage)
  CS_scaled = (CS-LB)/(1-LB)
  score = beta*npmi + (1-beta)*CS_scaled
  the Term and Morph of pairs with score > Th enter TermList   (second stage)
  every line of the base synonym file is filtered against TermList,
  and a line is written out when more than one term survives
"""
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
OUT = Path(__file__).parent / "out"
TFD = Path(r"SET_TFD_HOME_HERE")
BASE_TAG = {"KStem": "SynonymKStem", "SnowballEng": "SynonymSnowballEng"}


def prune(scores_csv, method, beta, th):
    """-> (mask of surviving pairs, TermList, lower bound)"""
    df = pd.read_csv(scores_csv)
    cs = df[method].to_numpy(float)
    npmi = df["npmi"].to_numpy(float)
    m = np.isfinite(cs)
    cs = np.where(m, cs, np.nanmedian(cs))
    q1, q3 = np.quantile(cs[m], [.25, .75])
    lb = q1 - 1.5 * (q3 - q1)
    keep1 = cs > lb
    scaled = (cs - lb) / (1 - lb)
    score = beta * npmi + (1 - beta) * scaled
    keep = keep1 & (score > th)
    terms = set(df.loc[keep, "Term"].astype(str)) | set(df.loc[keep, "Morph"].astype(str))
    return keep, terms, lb


def write_synonyms(coll, stem, terms, out_path):
    """Filters the base conflation set against TermList."""
    base = TFD / coll / f"{BASE_TAG[stem]}.txt"
    lines_out = []
    for line in open(base, encoding="utf-8"):
        kept = [w.strip() for w in line.strip().split(",") if w.strip() in terms]
        if len(kept) > 1:
            lines_out.append(",".join(kept))
    out_path.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    return len(lines_out)


def read_syn(p):
    return [l.strip() for l in open(p, encoding="utf-8") if l.strip()]


if __name__ == "__main__":
    print("=" * 90)
    print("CHECK: does cls_orig @ (beta=0.5, Th=0.6) reproduce the existing files?")
    print("=" * 90)
    tmp = OUT / "tmp_syn"
    tmp.mkdir(exist_ok=True)
    for coll, stem in [("CW09B", "KStem"), ("CW09B", "SnowballEng"),
                       ("NTCIR", "KStem"), ("NTCIR", "SnowballEng")]:
        sc = OUT / f"scores_{coll}_{stem}.csv"
        if not sc.exists():
            print(f"  {coll}/{stem}: no score file, skipped")
            continue
        keep, terms, lb = prune(sc, "cls_orig", 0.5, 0.6)
        p = tmp / f"{BASE_TAG[stem]}_bert-base-uncased_alpha0.5_threshold0.6.txt"
        n = write_synonyms(coll, stem, terms, p)

        ref = TFD / coll / "synonyms_backup" / p.name
        if not ref.exists():
            ref = TFD / coll / "Synonym_backup" / p.name
        if ref.exists():
            a, b = read_syn(p), read_syn(ref)
            sa = {frozenset(x.split(",")) for x in a}
            sb = {frozenset(x.split(",")) for x in b}
            same = "SAME" if sa == sb else "DIFFERENT"
            print(f"  {coll}/{stem:12s} IQR_LB={lb:.4f} kept_pairs={int(keep.sum()):5d} "
                  f"lines: produced={n:4d} existing={len(b):4d}  -> {same}")
            if sa != sb:
                print(f"      only in produced: {len(sa-sb)}   only in existing: {len(sb-sa)}")
        else:
            print(f"  {coll}/{stem:12s} no reference file ({p.name}) - produced n={n}")
