# Runs directory

Placeholder. Retrieval output is not tracked in the repository: the grid alone
produces several thousand run files, and each one lists 1000 documents per
query.

The drivers do not write here. They write into `TFD_HOME`, because the
evaluation scripts and the Java code both resolve paths through it:

```
TFD_HOME/<collection>/<runsPath>/<Tag>/<track>/<model>_contents_<Tag>_OR_all.txt
```

`runsPath` identifies the experiment, and the prefix says which driver produced
it:

| Prefix | Produced by | Example |
|---|---|---|
| `B_` | `run_baselines.ps1` | `B_KStem`, `B_QBS-Porter` |
| `W_` | `run_weighted.ps1`  | `W_QBS_KStem`, `W_MLFN_SnowballEng` |
| `X_` | `run_all.ps1`       | `X_subword_mean_KStem` |
| `G_` | `run_grid.ps1`      | `G_SynonymKStem_G_b0.5_t0.6` |

Each file is in TREC format, tab separated:

```
<qid>	Q0	<docid>	<rank>	<score>	<run tag>
```

Use this directory if you prefer to keep a copy of the runs alongside the code.
`.gitignore` excludes its contents but keeps this file.
