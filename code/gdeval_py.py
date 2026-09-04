"""Python equivalent of gdeval.pl -- exactly the same formulas.

gdeval.pl (TREC Web Track):
  gain      : only judgments > 0 are kept from the qrels; the rest count as 0
  DCG@k     : sum_{i=0}^{k-1} (2^g_i - 1) / log2(i+2)
  ideal     : DCG@k over the topic's positive judgments sorted in DESCENDING order
  nDCG@k    : DCG@k / ideal
  ERR@k     : r_i = (2^g_i - 1) / 2^MAX_JUDGMENT (=16);
              score += r_i * decay / (i+1);  decay *= (1 - r_i)
  ordering  : within a run, score DESCENDING, ties broken by docno ASCENDING (string)
  topic     : skipped entirely when it has no positive judgment

Spawning a perl subprocess for 1416 run files takes about two hours; this version
runs in minutes. Correctness is checked against gdeval.pl by `verify_gdeval.py`.
"""
import math
from collections import defaultdict

MAX_JUDGMENT = 4
_LOG2 = math.log(2.0)


def load_qrels(path):
    """-> judgments {topic: {docno: gain}}"""
    jud = defaultdict(dict)
    for line in open(path):
        p = line.split()
        if len(p) < 4:
            continue
        topic, docno, j = p[0], p[2], int(p[3])
        if j > 0:                       # gdeval yalnizca pozitifleri saklar
            jud[topic][docno] = j
    return dict(jud)


def _dcg(gains, k):
    s = 0.0
    for i in range(min(k, len(gains))):
        s += (2.0 ** gains[i] - 1.0) / (math.log(i + 2) / _LOG2)
    return s


def _err(gains, k):
    s, decay = 0.0, 1.0
    for i in range(min(k, len(gains))):
        r = (2.0 ** gains[i] - 1.0) / (2.0 ** MAX_JUDGMENT)
        s += r * decay / (i + 1)
        decay *= (1.0 - r)
    return s


def ideal_dcg(jud, k):
    return {t: _dcg(sorted(d.values(), reverse=True), k) for t, d in jud.items()}


def load_run(path):
    """-> {topic: [(score, docno), ...]}  (siralanmamis)"""
    run = defaultdict(list)
    for line in open(path):
        p = line.split()
        if len(p) < 5:
            continue
        run[p[0]].append((float(p[4]), p[2]))
    return run


def evaluate(run_path, jud, ideal, k=10):
    """-> {topic: (ndcg@k, err@k)}   -- yalnizca qrels'te pozitif yargisi olan topic'ler"""
    run = load_run(run_path)
    out = {}
    for topic, docs in run.items():
        if topic not in jud:            # seen{topic} yoksa gdeval atlar
            continue
        # gdeval runOrder: skor AZALAN; esitlikte docno AZALAN
        # (perl'de  $docnoA lt $docnoB -> return 1,  yani kucuk docno SONRA gelir)
        docs.sort(key=lambda x: x[1], reverse=True)   # once docno azalan
        docs.sort(key=lambda x: -x[0])                # sonra skor azalan (kararli)
        g = [jud[topic].get(d, 0) for _, d in docs[:k]]
        idl = ideal[topic]
        out[topic] = (_dcg(g, k) / idl if idl > 0 else 0.0, _err(g, k))
    return out
