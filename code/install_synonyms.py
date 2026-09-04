"""Copies the synonym files of this repository into TFD_HOME under the names the
retrieval drivers expect.

The repository stores the files under readable names:

    synonyms/NTCIR/proposed/SynonymPorter_b0.5_t0.6.txt

while the drivers (RunBatch, run_all.ps1, run_baselines.ps1, ...) look them up
inside the collection directory of TFD_HOME under their internal names:

    TFD_HOME/NTCIR/SynonymSnowballEng_G_b0.5_t0.6.txt

This script performs exactly that mapping. Nothing is overwritten unless
--force is given, and nothing is copied at all until --write is given.

Usage:
    python install_synonyms.py                 # dry run: only reports
    python install_synonyms.py --write         # actually copy
    python install_synonyms.py --write --force # copy over existing files
"""
import shutil, sys
from pathlib import Path

REPO = Path(__file__).parent.parent
TFD = Path(r"SET_TFD_HOME_HERE")

# repository directory name -> TFD_HOME collection directory name
COLL = {"CW09B": "CW09B", "NTCIR": "NTCIR"}

# repository relative path -> file name inside TFD_HOME/<collection>/
MAP = {
    "base/SynonymKStem.txt":            "SynonymKStem.txt",
    "base/SynonymPorter.txt":           "SynonymSnowballEng.txt",
    "baselines/SynonymHPS.txt":         "SynonymHPS.txt",
    "baselines/SynonymKStem_MLFN.txt":  "SynonymKStem_MLFN_sel.txt",
    "baselines/SynonymPorter_MLFN.txt": "SynonymSnowballEng_MLFN_sel.txt",
    "baselines/npmi_pred_KStem.txt":    "SynonymKStemMLFN.txt",
    "baselines/npmi_pred_Porter.txt":   "SynonymSnowballEngMLFN.txt",
    "proposed/SynonymKStem_b0.5_t0.6.txt":  "SynonymKStem_G_b0.5_t0.6.txt",
    "proposed/SynonymPorter_b0.5_t0.6.txt": "SynonymSnowballEng_G_b0.5_t0.6.txt",
}
for m in ["subword_mean", "cls_masked", "bert_static", "sbert", "fasttext", "query_ctx"]:
    MAP[f"representations/SynonymKStem_{m}.txt"] = f"SynonymKStem_X_{m}.txt"
    MAP[f"representations/SynonymPorter_{m}.txt"] = f"SynonymSnowballEng_X_{m}.txt"


def main():
    write = "--write" in sys.argv
    force = "--force" in sys.argv
    if str(TFD).startswith("SET_"):
        raise SystemExit("Set TFD at the top of this file to your TFD_HOME first.")

    n_ok = n_var = n_yok = 0
    for repo_coll, tfd_coll in COLL.items():
        hedef_dir = TFD / tfd_coll
        if not hedef_dir.exists():
            print(f"  {tfd_coll}: no such directory in TFD_HOME, skipped")
            continue
        for rel, ad in MAP.items():
            src = REPO / "synonyms" / repo_coll / rel
            dst = hedef_dir / ad
            if not src.exists():
                print(f"  MISSING  {repo_coll}/{rel}")
                n_yok += 1
                continue
            if dst.exists() and not force:
                n_var += 1
                continue
            if write:
                shutil.copy2(src, dst)
            n_ok += 1
            print(f"  {'copied ' if write else 'would copy'}  {repo_coll}/{rel}"
                  f"  ->  {tfd_coll}/{ad}")

    print(f"\n{n_ok} file(s) {'copied' if write else 'to copy'}, "
          f"{n_var} already present (use --force to overwrite), {n_yok} missing")
    if not write:
        print("dry run: nothing was written. Add --write to apply.")


if __name__ == "__main__":
    main()
