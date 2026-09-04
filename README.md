# Pruned Conflation Sets

Code and data for **"Enhancing Web Search Effectiveness and Robustness Using
BERT- and Co-occurrence-Based Pruned Conflation Sets"**.

A stemmer groups morphological variants into conflation sets. Some of those
groupings help retrieval and some hurt it. This work prunes the sets before they
are ever used: each (term, variant) pair is scored with a contextual similarity
and a corpus co-occurrence statistic, and pairs that fail both tests are removed.
The result is a much smaller conflation set that retrieves better than the
stemmer it came from.

The pruning has two stages.

1. **Similarity floor.** On the contextual similarity `CS`, the interquartile
   rule gives a lower bound `LB = Q1 - 1.5 * (Q3 - Q1)`. Pairs with `CS <= LB`
   are dropped, and the survivors are rescaled to `CS_scaled = (CS - LB) / (1 - LB)`.
2. **Combined threshold.** A pair is kept when
   `S = beta * NPMI + (1 - beta) * CS_scaled > Th`.

The operating point used throughout the paper is **beta = 0.5, Th = 0.6**. At
that point 97.6% of the 28,814 scored pairs are removed, which leaves roughly a
quarter of the conflation lines standing:

| Collection | Stemmer | Pairs kept | Lines: base -> pruned |
|---|---|---|---|
| CW09B | KStem  | 120 / 6,908 | 409 -> 102 |
| CW09B | Porter | 204 / 8,249 | 415 -> 136 |
| NTCIR | KStem  | 138 / 5,956 | 380 -> 104 |
| NTCIR | Porter | 239 / 7,701 | 386 -> 141 |

What this buys at query time is a much smaller expansion. The full conflation
set expands a query by a factor of between 16.7 and 20.8; the pruned set expands
it by 1.25 to 1.54, a 93.5% reduction in the terms actually submitted.

## Layout

```
code/          every script, flat, so that "out/" always sits next to them
  java/        the retrieval drivers and the two Lucene classes they need
  out/         the four score tables (see "Where NPMI comes from" below)
synonyms/      the conflation sets used in the paper (42 files, 6.1 MB)
  CW09B/       ClueWeb09 Category B
  NTCIR/       ClueWeb12-B13, retrieved for the NTCIR WWW tracks
    base/            the unpruned sets produced by KStem and Porter
    proposed/        the pruned sets at beta=0.5, Th=0.6
    representations/ the pruned sets of the six representations compared in Table 2
    baselines/       HPS, and the MLFN selection and prediction files
index/         placeholder; see index/README.md
runs/          placeholder for retrieval output
```

`Porter` in the file names is the Snowball English stemmer, which the retrieval
toolkit calls `SnowballEng`. The two names refer to the same stemmer.

## Requirements

Python 3.12 (the version used for the reported runs).

```bash
pip install -r requirements.txt
```

Retrieval additionally needs a JDK and the
`lucene-clueweb-retrieval` toolkit, which supplies `Searcher`, `Tag`, `DataSet`
and the similarity implementations: <https://github.com/iorixxx/lucene-clueweb-retrieval>.

`eval_baselines.py` and `evaluate_runs.py` shell out to `gdeval.pl` and so need
Perl. The other two evaluation scripts use the bundled `gdeval_py.py` instead,
which is a pure-Python reimplementation; it was checked against `gdeval.pl` on
492 of 492 query-run pairs and agrees exactly.

## Paths to set

Every machine-specific path is a placeholder of the form `SET_..._HERE`. There
are three:

| Placeholder | Meaning | Files |
|---|---|---|
| `SET_TFD_HOME_HERE` | the `TFD_HOME` root (index, conflation sets, qrels) | 7 Python scripts, `$TFD` in the 4 `.ps1` |
| `SET_LUCENE_TOOLKIT_DIR_HERE` | the built `lucene-clueweb-retrieval` distribution | `$DIST` in the 4 `.ps1` |
| `SET_NPMI_WORKBOOK_DIR_HERE` | the directory holding the `npmi-*.xlsx` workbooks | `score_all.py`, `fasttext_extract.py` |

`fasttext_extract.py` also has `SET_FASTTEXT_CACHE_DIR_HERE`, which is only the
directory to keep the downloaded `cc.en.300.bin` in. Every script writes its
output to `code/out/`, which needs no configuration.

`index/README.md` describes the `TFD_HOME` layout the drivers expect.

## Where NPMI comes from

The method combines two signals, and they reach the code by different routes.
This matters, so it is worth stating plainly: **no script in this repository
computes NPMI.**

**The NPMI the pruning uses** — the `npmi` term in
`S = beta * NPMI + (1 - beta) * CS_scaled` — is read, not computed.
`score_all.py` takes it from the workbooks `npmi-CW09B.xlsx` and
`npmi-NTCIR-Bert.xlsx` (columns `QID, Term, Morph, npmi`), which were produced
by an earlier stage of this project and are not distributed here. The values are
pointwise mutual information, normalised, computed over the same unstemmed index
the retrieval runs against, using the `PMI` class of the
`lucene-clueweb-retrieval` toolkit (`edu.anadolu.qpp.PMI.npmi(a, b)`).

So that the pruning stage is reproducible without those workbooks, the four
score tables are shipped under `code/out/`:

```
scores_CW09B_KStem.csv   scores_CW09B_SnowballEng.csv
scores_NTCIR_KStem.csv   scores_NTCIR_SnowballEng.csv
```

Each row is one (query term, morphological variant) pair with its `npmi` value
and all six similarity columns. With these in place the whole pruning stage runs
with no index, no workbooks and no BERT. Point `SET_TFD_HOME_HERE` at any
directory containing empty `CW09B/` and `NTCIR/` folders, then:

```bash
cd code
python install_synonyms.py --write   # puts the base conflation sets in place
python make_synonyms.py              # prunes at beta=0.5, Th=0.6
python gen_grid.py                   # the full 11x11 grid + grid_manifest.json
```

On a clean checkout this produces 149 and 253 conflation lines for CW09B
(KStem and Porter), 153 and 233 for ClueWeb12-B13, and 360 unique grid
configurations out of 484. Retrieval is the first step that needs the index.

Re-running `score_all.py` is only necessary to regenerate the similarity columns
from scratch, and that is the step that needs the workbooks.

**The NPMI the weighting baselines use** is a different quantity: it is computed
from the index at query time by `SynonymWeightFunctions.calculateNPMI` in the
toolkit, over pairs of query terms rather than (term, variant) pairs.
`code/java/DumpNpmi.java` precomputes that table once and caches it, which is
what makes the QBS runs tractable. MLFN replaces those values with predicted
ones, shipped here as `synonyms/*/baselines/npmi_pred_*.txt`.

## Running the experiments

### 1. Scoring

Scores every (term, variant) pair under all six representations, for both
stemmers and both collections.

```bash
cd code
python score_all.py
```

This writes `out/scores_<collection>_<stemmer>.csv`, one row per pair, with a
column per representation plus the `npmi` column read from the workbooks. It
downloads `bert-base-uncased` and the sentence-transformers model on first use.
The `fasttext` column needs `out/fasttext_vectors.npz`, which comes from a
separate one-off step because the fastText model needs about 7 GB of RAM:

```bash
python fasttext_extract.py
```

To run the four (collection, stemmer) jobs in parallel, pass a subset:
`python score_all.py 0,1` and `python score_all.py 2,3`.

### 2. Pruning

`make_synonyms.py` holds the pruning itself. Running it directly performs a
self-check: it re-derives the shipped sets and diffs them against the files in
`TFD_HOME`.

```bash
python make_synonyms.py
```

`gen_grid.py` generates the conflation sets for the whole 11x11 `(beta, Th)`
grid. Files with identical content are de-duplicated, since at high thresholds
many configurations collapse onto the same (often empty) set. It writes the
files into `TFD_HOME` and records the mapping in `out/grid_manifest.json`.

```bash
python gen_grid.py
```

### 3. Installing the shipped conflation sets

If you would rather use the exact sets from the paper than regenerate them,
copy them into `TFD_HOME` under the names the drivers expect:

```bash
python install_synonyms.py            # dry run, reports what it would copy
python install_synonyms.py --write
```

### 4. Retrieval

Needs the index. Run from `code/` in PowerShell.

```powershell
. .\env.ps1                                          # optional: keeps caches off the system drive

.\run_baselines.ps1 -Collection CW09B                # NoStem, Porter, KStem, HPS, QBS-*
.\run_weighted.ps1  -Collection CW09B -System QBS    # QBS   (full NPMI)
.\run_weighted.ps1  -Collection CW09B -System MLFN   # MLFN  (predicted NPMI)
.\run_all.ps1       -Collection CW09B                # the six representations
.\run_grid.ps1      -Collection CW09B                # the 11x11 grid
```

Repeat with `-Collection NTCIR` for ClueWeb12-B13. Every driver takes
`-Parallel`, `-Threads` and `-Models`; the paper uses
`-Models "BM25k1.2b0.75,DPH"`. Each one prints an output check at the end, and
an empty run file always means a failed run.

The Java sources under `code/java/` are compiled against the toolkit:

```bash
javac -encoding UTF-8 -cp "<toolkit>/repo/*;<toolkit>/FrequencyDistributionAnalysis-8.0.jar" \
      java/RunBatch.java java/RunMlfn.java java/DumpNpmi.java \
      java/org/apache/lucene/search/SumRoundSynonymQuery.java \
      java/org/apache/lucene/search/MlfnWeightFunctions.java
```

`RunBatch` runs an ordinary conflation-set system. `RunMlfn` runs the weighted
systems; QBS and MLFN share it and differ only in where NPMI comes from, which
is what makes that comparison controlled. `DumpNpmi` precomputes the NPMI table
those weighted systems need, which takes the QBS runs from about nine hours to
roughly a fifth of that:

```bash
java -cp "<toolkit>/repo/*;<toolkit>/FrequencyDistributionAnalysis-8.0.jar;." \
     DumpNpmi <TFD_HOME> CW09B SynonymKStemQBS SynonymKStem.txt npmi_cache.csv
```

### 5. Evaluation: nDCG@10, ERR@10, MAP

nDCG@10 and ERR@10 follow `gdeval.pl`; MAP is computed over the top 1000 with
`pytrec_eval`, matching `trec_eval`. Each script writes a per-query CSV, so any
number in the paper can be traced back to individual queries.

```bash
python eval_baselines.py CW09B     # -> out/baseline_perquery_CW09B.csv
python eval_weighted.py            # -> out/weighted_perquery.csv  (both collections)
python evaluate_runs.py CW09B      # -> out/eval_perquery_CW09B.csv   (representations)
python eval_grid.py CW09B          # -> out/grid_perquery_CW09B.parquet
```

`eval_baselines.py` also prints its numbers side by side with Table 3 of the
paper. The baselines do not depend on the BERT scoring at all, so that
comparison is a check on the whole pipeline: if the baselines reproduce, the
index, the runs and the evaluation are all behaving.

`gdeval_py.py` can also be used on its own:

```python
from gdeval_py import load_qrels, ideal_dcg, evaluate
qrels = load_qrels("qrels.web.1-50.txt")
scores = evaluate("run.txt", qrels, ideal_dcg(qrels, 10), 10)   # {qid: (ndcg, err)}
```

### 6. Tables and analysis

```bash
python main_table.py           # main results   -> out/main_table.csv
python robustness.py           # TRisk          -> out/trisk.csv, out/trisk_track.csv
python significance_main.py    # significance   -> out/significance_main.csv
```

`main_table.py` reports both selection protocols: the leave-one-query-out
protocol, and the single fixed configuration `beta=0.5, Th=0.6` applied
unchanged everywhere.

`robustness.py` is a port of the TRisk measure of Dincer et al.: losses against
the baseline are inflated by `(alpha + 1)` before the t-statistic is formed, so
increasing `alpha` charges more for hurting a query. At `alpha = 0` it reduces
to the ordinary paired t-statistic. The baseline is NoStem throughout, as in the
paper.

`significance_main.py` pairs systems per query, runs a permutation (sign) test
with 10,000 replicates alongside the paired t-test, and applies a
Holm-Bonferroni correction within each family of comparisons.

## What is not included

- **The Lucene indexes.** About 30 GB per collection, and built from licensed
  document collections. See `index/README.md`.
- **Topics and relevance judgments.** Available from NIST (TREC Web) and the
  NTCIR organisers; `index/README.md` lists the file names the scripts expect.
- **The NPMI workbooks.** `score_all.py` reads the (term, variant) pairs and
  their NPMI values from `npmi-*.xlsx`. See "Where NPMI comes from" above; the
  four shipped score tables make this stage reproducible without them.
- **The fastText model.** `cc.en.300.bin` from
  <https://fasttext.cc/docs/en/crawl-vectors.html>.

## A note on the `[CLS]` representation

The version of this work that was first submitted used a `[CLS]` pooling
implementation with a defect: the two words of a pair were tokenised as one
batch, padded to a common length, and the attention mask was not passed to the
model. A term's vector therefore depended on the length of whichever variant it
happened to be paired with. `scorers.py` keeps that code as `cls_orig` so the
original numbers stay reproducible, and adds `cls_masked`, which is the same
pooling with the mask passed. The paper reports `cls_masked`, and the shipped
conflation sets under `representations/` use it.
