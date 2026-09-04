"""Loads cc.en.300.bin once, extracts the vectors of every word that occurs in
ALL collections, saves them as .npz, then releases the model.

The model needs ~7 GB of RAM, so it is run once and every later job reads from
the small .npz file instead.
"""
import gzip, shutil, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
PMV = Path(__file__).parent.parent / "npmi"
CACHE = Path(r"SET_FASTTEXT_CACHE_DIR_HERE")
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
GZ, BIN = CACHE / "cc.en.300.bin.gz", CACHE / "cc.en.300.bin"

# --- 1) collect every word
SOURCES = ["npmi_CW09B_SnowballEng.csv", "npmi_CW09B_KStem.csv",
           "npmi_NTCIR_SnowballEng.csv", "npmi_NTCIR_KStem.csv"]
words = set()
for f in SOURCES:
    p = PMV / f
    if not p.exists():
        print(f"  skipped (missing): {f}")
        continue
    d = pd.read_csv(p)
    words |= set(d["Term"].astype(str)) | set(d["Morph"].astype(str))
    print(f"  {f}: {len(d)} rows -> {len(words)} unique words so far")
words = sorted(words)
print(f"\nTOTAL unique words: {len(words)}")

# --- 2) open the model (decompress it first when needed)
if not BIN.exists():
    if not GZ.exists():
        raise SystemExit(f"model not found: {GZ}")
    print(f"\ndecompressing: {GZ.name} -> {BIN.name} ...")
    with gzip.open(GZ, "rb") as fi, open(BIN, "wb") as fo:
        shutil.copyfileobj(fi, fo, length=1 << 24)
    print(f"done: {BIN.stat().st_size/1e9:.1f} GB")

print("\nloading the fastText model (~7 GB RAM, a few minutes)...")
from gensim.models.fasttext import load_facebook_vectors
kv = load_facebook_vectors(str(BIN))
print(f"loaded. vocabulary: {len(kv.key_to_index)}  dimension: {kv.vector_size}")

# --- 3) extract the vectors
in_vocab = np.array([w in kv.key_to_index for w in words])
print(f"in vocabulary : {int(in_vocab.sum())}/{len(words)} "
      f"({100*in_vocab.mean():.1f}%)  -- the rest is composed from character n-grams")

V = np.vstack([kv[w] for w in words]).astype(np.float32)
np.savez_compressed(OUT / "fasttext_vectors.npz",
                    words=np.array(words, dtype=object),
                    vectors=V, in_vocab=in_vocab)
print(f"\n-> out/fasttext_vectors.npz  ({V.shape})")
print("the model can be released now; later jobs read from this file.")
