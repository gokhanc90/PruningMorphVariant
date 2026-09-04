# Index directory

The Lucene indexes cannot be distributed. The ClueWeb09 and ClueWeb12 document
collections are licensed by Carnegie Mellon University, and a built index is
around **30 GB per collection**. This directory is a placeholder that documents
what has to exist here before the retrieval scripts can run.

Everything in this repository *except retrieval* runs without an index. Scoring,
pruning, the grid, and every analysis script work from the CSV files under
`code/out/`. Only `run_*.ps1` and the two Java drivers need the index.

## What the drivers expect

The retrieval toolkit resolves the index through `TFD_HOME`, so the layout is
fixed:

```
TFD_HOME/
  CW09B/                       ClueWeb09 Category B
    indexes/
      NoStem/                  <- the Lucene index directory
    SynonymKStem.txt           conflation sets (installed by install_synonyms.py)
    ...
  NTCIR/                       ClueWeb12-B13
    indexes/
      NoStem/
    ...
  topics-and-qrels/            topics and relevance judgments
```

`TFD_HOME` is the path you set at the top of the Python scripts
(`SET_TFD_HOME_HERE`) and in the `$TFD` variable of the PowerShell drivers.

## Why a single unstemmed index

There is exactly one index per collection, built with `Tag.NoStem`, and it is
shared by every system in the paper. Stemming is never applied at indexing time.
A conflation set is applied at **query time** through Lucene's
`SynonymGraphFilterFactory`: a query term is expanded into its morphological
variants, and the variants are matched against the unstemmed index.

This is what makes the comparison controlled. NoStem, Porter, KStem, HPS, QBS,
MLFN and the pruned systems all read the same postings; they differ only in
which variants a query term expands to, and how those variants are weighted.
Nothing needs to be re-indexed when a conflation set changes, which is also why
the 11x11 grid is feasible at all.

## Building the index

The indexes were built with the `lucene-clueweb-retrieval` toolkit
(<https://github.com/iorixxx/lucene-clueweb-retrieval>), the same toolkit that
supplies `Searcher`, `Tag`, `DataSet` and the similarity implementations used by
`code/java/RunBatch.java`. Follow that project's indexing instructions and build
with the `NoStem` tag; the drivers here need nothing else from it.

Two collections are used in the paper:

| Directory | Collection | Tracks | Topics |
|---|---|---|---|
| `CW09B` | ClueWeb09 Category B | TREC Web 2009-2012 (WT09-WT12) | 197 judged (49/48/50/50) |
| `NTCIR` | ClueWeb12-B13 | NTCIR-13 WWW-1, NTCIR-14 WWW-2 | 180 (100/80) |

## Topics and relevance judgments

These are not distributed here either. TREC Web track qrels are available from
NIST (<https://trec.nist.gov/data/webmain.html>) and the NTCIR WWW qrels from
the NTCIR organisers (<https://research.nii.ac.jp/ntcir/>). Place them under
`TFD_HOME/topics-and-qrels/` using the names the evaluation scripts look up:

```
qrels.web.1-50.txt      WT09        qrels.www.1-100.txt     NTCIR-13 (WWW-1)
qrels.web.51-100.txt    WT10        qrels.www.101-180.txt   NTCIR-14 (WWW-2)
qrels.web.101-150.txt   WT11
qrels.web.151-200.txt   WT12
```

## Checking the setup

Once the index is in place, the quickest check is a single baseline run:

```powershell
.\run_baselines.ps1 -Collection CW09B -Only NoStem
```

It writes to `TFD_HOME/CW09B/B_NoStem/` and prints an output check at the end.
Empty run files mean the index was not found or was not readable.
