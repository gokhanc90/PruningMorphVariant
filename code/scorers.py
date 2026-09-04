"""Contextual-similarity scorers for a (query term, morphological variant) pair.

Methods
-------
cls_orig      : The submitted version. The two words are batched TOGETHER, padded
                with zeros ([PAD]) to a common length, attention_mask is NOT passed,
                and the [CLS] of the final layer is read.
                -> a term's vector depends on the length of the variant it happens
                   to be paired with (the defect).
cls_masked    : The same, but attention_mask is passed; the vector is specific to
                the word (the minimal correction).
subword_mean  : Mean over the word's OWN sub-word tokens (special tokens excluded).
query_ctx     : The term is encoded inside its own query; the variant is substituted
                in its place in the same query. In both cases the sub-words of the
                word in question are averaged.
sbert         : sentence-transformers model (isolated word).
fasttext      : character n-gram based static embedding (composes OOV from sub-words).
"""
import numpy as np
import torch


def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


class BertScorer:
    def __init__(self, name="bert-base-uncased"):
        from transformers import AutoTokenizer, AutoModel
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name)
        self.model.eval()

    # ---------- the submitted version ----------
    def cls_orig(self, w1, w2):
        ids = []
        for w in (w1, w2):
            e = self.tok(w, add_special_tokens=True, max_length=16,
                         padding="longest", return_tensors="pt", truncation=True)
            ids.append(e["input_ids"])
        m = max(i.shape[1] for i in ids)
        t = torch.cat([torch.nn.ZeroPad2d((0, m - i.shape[1], 0, 0))(i) for i in ids], 0)
        with torch.no_grad():
            v = self.model(t).last_hidden_state[:, 0, :].numpy()
        return _cos(v[0], v[1])

    # ---------- word-specific vectors (batched) ----------
    def _encode(self, words, bs=256):
        cls_out, sub_out = [], []
        for i in range(0, len(words), bs):
            chunk = [str(w) for w in words[i:i + bs]]
            enc = self.tok(chunk, add_special_tokens=True, max_length=16,
                           padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                h = self.model(**enc).last_hidden_state          # (B,T,H)
            cls_out.append(h[:, 0, :].numpy())
            # exclude special tokens and padding -> the word's own sub-words
            mask = enc["attention_mask"].clone()
            mask[:, 0] = 0                                        # [CLS]
            idx = enc["attention_mask"].sum(1) - 1
            mask[torch.arange(mask.size(0)), idx] = 0             # [SEP]
            m = mask.unsqueeze(-1).float()
            sub_out.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).numpy())
        return np.vstack(cls_out), np.vstack(sub_out)

    def word_vectors(self, words):
        """-> (cls_masked, subword_mean) dictionaries"""
        words = [str(w) for w in words]
        c, s = self._encode(words)
        return ({w: v for w, v in zip(words, c)},
                {w: v for w, v in zip(words, s)})

    def static_vectors(self, words):
        """Sub-word mean over BERT's input (WordPiece) embedding table.
        No context at all -- an honest 'static' baseline."""
        emb = self.model.get_input_embeddings().weight.detach().numpy()
        out = {}
        for w in words:
            w = str(w)
            ids = self.tok.convert_tokens_to_ids(self.tok.tokenize(w))
            out[w] = emb[ids].mean(0) if ids else np.zeros(emb.shape[1])
        return out

    # ---------- query context ----------
    def query_context_vec(self, query, target, bs=None):
        """Sub-word mean of the word `target` inside the query.
        Returns None when target does not occur in the query."""
        words = str(query).split()
        low = [w.lower() for w in words]
        t = str(target).lower()
        if t not in low:
            return None
        pos = low.index(t)
        enc = self.tok(words, is_split_into_words=True, add_special_tokens=True,
                       truncation=True, max_length=64, return_tensors="pt")
        wids = enc.word_ids(0)
        sel = [i for i, w in enumerate(wids) if w == pos]
        if not sel:
            return None
        with torch.no_grad():
            h = self.model(**enc).last_hidden_state[0].numpy()
        return h[sel].mean(0)

    # ---------- batched query context ----------
    def _pooled_batch(self, sentences, positions, bs=64):
        """sentences: [[w1,w2,...], ...]  positions: index of the target word.
        Returns the mean of the target word's sub-words in each sentence."""
        out = np.zeros((len(sentences), self.model.config.hidden_size), dtype=np.float32)
        for s in range(0, len(sentences), bs):
            chunk = sentences[s:s + bs]
            pos = positions[s:s + bs]
            enc = self.tok(chunk, is_split_into_words=True, add_special_tokens=True,
                           truncation=True, max_length=64, padding=True,
                           return_tensors="pt")
            with torch.no_grad():
                h = self.model(**enc).last_hidden_state.numpy()
            for r in range(len(chunk)):
                wids = enc.word_ids(r)
                sel = [i for i, w in enumerate(wids) if w == pos[r]]
                out[s + r] = h[r, sel].mean(0) if sel else np.nan
        return out

    def query_context_sims(self, items, bs=64):
        """items: [(query, term, variant), ...] -> array of cosines.
        The term's vector in its own query is computed once per (qid, term)."""
        n = len(items)
        out = np.full(n, np.nan)

        base_key, base_sent, base_pos, base_idx = {}, [], [], []
        var_sent, var_pos, var_idx = [], [], []
        for i, (q, term, var) in enumerate(items):
            words = str(q).split()
            low = [w.lower() for w in words]
            t = str(term).lower()
            if t not in low:
                continue
            p = low.index(t)
            k = (str(q), t)
            if k not in base_key:
                base_key[k] = len(base_sent)
                base_sent.append(words)
                base_pos.append(p)
            base_idx.append((i, base_key[k]))
            sw = list(words)
            sw[p] = str(var)
            var_sent.append(sw)
            var_pos.append(p)
            var_idx.append(i)

        if not var_sent:
            return out
        B = self._pooled_batch(base_sent, base_pos, bs)
        V = self._pooled_batch(var_sent, var_pos, bs)
        bmap = {i: j for i, j in base_idx}
        for k, i in enumerate(var_idx):
            a, b = B[bmap[i]], V[k]
            if np.isfinite(a).all() and np.isfinite(b).all():
                out[i] = _cos(a, b)
        return out

    def query_context_sim(self, query, term, variant):
        """The term in its own query; the variant substituted in its place."""
        v1 = self.query_context_vec(query, term)
        if v1 is None:
            return None
        words = str(query).split()
        low = [w.lower() for w in words]
        pos = low.index(str(term).lower())
        swapped = list(words)
        swapped[pos] = str(variant)
        v2 = self.query_context_vec(" ".join(swapped), variant)
        if v2 is None:
            return None
        return _cos(v1, v2)


class SbertScorer:
    def __init__(self, name="sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(name)

    def word_vectors(self, words):
        words = [str(w) for w in words]
        v = self.m.encode(words, batch_size=256, show_progress_bar=False,
                          convert_to_numpy=True)
        return {w: x for w, x in zip(words, v)}


class FastTextScorer:
    def __init__(self, bin_path):
        import fasttext
        self.m = fasttext.load_model(str(bin_path))

    def word_vectors(self, words):
        return {str(w): self.m.get_word_vector(str(w)) for w in words}


def cosine_from_vectors(vecs, pairs):
    """pairs: [(w1,w2), ...] -> array of cosines (NaN for a missing word)."""
    out = np.full(len(pairs), np.nan)
    norm = {w: v / (np.linalg.norm(v) + 1e-12) for w, v in vecs.items()}
    for i, (a, b) in enumerate(pairs):
        a, b = str(a), str(b)
        if a in norm and b in norm:
            out[i] = float(norm[a] @ norm[b])
    return out
