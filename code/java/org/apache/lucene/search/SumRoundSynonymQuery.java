/*
 * A line-by-line copy of SynonymWeightedQuery -- the ONLY difference is in tf().
 *
 * Original (round per variant):
 *     tf += Math.round(freq_i * factor_i);
 * This version (sum first, then round):
 *     sum += freq_i * factor_i;   ...   return round(sum);
 *
 * Why: under the original scheme a single occurrence of a variant weighted below
 * 0.5 rounds to 0 and is discarded outright, so variants cannot build up evidence
 * TOGETHER. With the narrow weights MLFN produces (0.21-0.32) no variant can
 * contribute from a single occurrence at all, which penalises the method for a
 * reason that belongs to the implementation rather than to the method.
 *
 * We did not move to a fully fractional tf: DPH, DLH13, PL2 and DirichletLM
 * produce NEGATIVE scores in the tf<0.15 region (see ProbeFractionalTF). Keeping
 * tf an integer leaves those models in the regime they were tested in.
 */
package org.apache.lucene.search;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Set;

import edu.anadolu.analysis.Analyzers;
import edu.anadolu.analysis.Tag;
import edu.anadolu.datasets.DataSet;
import edu.anadolu.qpp.Commonality;
import org.apache.lucene.index.*;
import org.apache.lucene.search.similarities.Similarity;
import org.apache.lucene.search.similarities.Similarity.SimScorer;
import org.apache.lucene.util.BytesRef;

public final class SumRoundSynonymQuery extends Query {
    private final Term[] terms;
    private final Term orginal;
    private final Term[] otherOrj;
    private final Tag tag;
    private final DataSet dataset;
    private SynonymWeightFunctions functions;
    Commonality com;

    public SumRoundSynonymQuery(SynonymWeightFunctions functions, DataSet dataset, Term[] otherOrj,
                                Tag tag, Term original, Commonality com, Term... terms) {
        this.terms = Objects.requireNonNull(terms).clone();
        this.orginal = original;
        this.otherOrj = otherOrj;
        this.functions = functions;
        this.tag = tag;
        this.dataset = dataset;
        this.com = com;

        if (tag.toString().contains("SynonymKStem"))
            com.setAnalyzer(Analyzers.analyzer(Tag.SynonymKStem, dataset.collectionPath()));
        else if (tag.toString().contains("SynonymSnowballEng"))
            com.setAnalyzer(Analyzers.analyzer(Tag.SynonymSnowballEng, dataset.collectionPath()));

        String field = null;
        for (Term term : terms) {
            if (field == null) field = term.field();
            else if (!term.field().equals(field))
                throw new IllegalArgumentException("Synonyms must be across the same field");
        }
        if (!original.field().equals(field))
            throw new IllegalArgumentException("Original term and Synonyms must be across the same field");
        if (terms.length > BooleanQuery.getMaxClauseCount())
            throw new BooleanQuery.TooManyClauses();
        Arrays.sort(this.terms);
    }

    public List<Term> getTerms() {
        return Collections.unmodifiableList(Arrays.asList(terms));
    }

    @Override
    public String toString(String field) {
        StringBuilder builder = new StringBuilder("SumRoundSynonym(");
        for (int i = 0; i < terms.length; i++) {
            if (i != 0) builder.append(" ");
            builder.append(new TermQuery(terms[i]).toString(field));
        }
        return builder.append(")").toString();
    }

    @Override
    public int hashCode() {
        return 31 * classHash() + Arrays.hashCode(terms);
    }

    @Override
    public boolean equals(Object other) {
        return sameClassAs(other) && Arrays.equals(terms, ((SumRoundSynonymQuery) other).terms);
    }

    @Override
    public Query rewrite(IndexReader reader) throws IOException {
        if (terms.length == 0) return new BooleanQuery.Builder().build();
        if (terms.length == 1) return new TermQuery(terms[0]);
        return this;
    }

    @Override
    public Weight createWeight(IndexSearcher searcher, boolean needsScores, float boost) throws IOException {
        if (needsScores) return new SynonymWeight(this, searcher, boost);
        BooleanQuery.Builder bq = new BooleanQuery.Builder();
        for (Term term : terms) bq.add(new TermQuery(term), BooleanClause.Occur.SHOULD);
        return searcher.rewrite(bq.build()).createWeight(searcher, needsScores, boost);
    }

    class SynonymWeight extends Weight {
        private final TermContext termContexts[];
        private final Similarity similarity;
        private final Similarity.SimWeight simWeight;
        final double factors[];

        SynonymWeight(Query query, IndexSearcher searcher, float boost) throws IOException {
            super(query);
            CollectionStatistics collectionStats = searcher.collectionStatistics(terms[0].field());
            long docFreq = 0;
            long totalTermFreq = 0;
            termContexts = new TermContext[terms.length];
            factors = new double[terms.length];

            for (int i = 0; i < termContexts.length; i++) {
                if (tag.toString().contains("QBS")) factors[i] = functions.QBSART(orginal, terms[i], otherOrj);
                else if (tag.toString().contains("BERT")) factors[i] = functions.BERTSim(orginal, terms[i]);
                else throw new RuntimeException("Weighting function is not found!");
                termContexts[i] = build(searcher.getTopReaderContext(), terms[i], factors[i]);
                TermStatistics termStats = searcher.termStatistics(terms[i], termContexts[i]);
                docFreq = Math.max(termStats.docFreq(), docFreq);
                if (termStats.totalTermFreq() == -1) totalTermFreq = -1;
                else if (totalTermFreq != -1) totalTermFreq += termStats.totalTermFreq();
            }

            docFreq = com.df(orginal.text());

            TermStatistics pseudoStats = new TermStatistics(null, docFreq, totalTermFreq);
            this.similarity = searcher.getSimilarity(true);
            this.simWeight = similarity.computeWeight(boost, collectionStats, pseudoStats);
        }

        @Override
        public void extractTerms(Set<Term> terms) {
            for (Term term : SumRoundSynonymQuery.this.terms) terms.add(term);
        }

        @Override
        public Matches matches(LeafReaderContext context, int doc) throws IOException {
            String field = terms[0].field();
            Terms terms = context.reader().terms(field);
            if (terms == null || terms.hasPositions() == false) return super.matches(context, doc);
            return MatchesUtils.forField(field, () -> DisjunctionMatchesIterator.fromTerms(
                    context, doc, getQuery(), field, Arrays.asList(SumRoundSynonymQuery.this.terms)));
        }

        @Override
        public Explanation explain(LeafReaderContext context, int doc) throws IOException {
            Scorer scorer = scorer(context);
            if (scorer != null) {
                int newDoc = scorer.iterator().advance(doc);
                if (newDoc == doc) {
                    final float freq;
                    if (scorer instanceof SynonymScorer) {
                        SynonymScorer synScorer = (SynonymScorer) scorer;
                        freq = synScorer.tf(synScorer.getSubMatches());
                    } else {
                        assert scorer instanceof TermScorer;
                        freq = ((TermScorer) scorer).freq();
                    }
                    SimScorer docScorer = similarity.simScorer(simWeight, context);
                    Explanation freqExplanation = Explanation.match(freq, "termFreq=" + freq);
                    Explanation scoreExplanation = docScorer.explain(doc, freqExplanation);
                    return Explanation.match(scoreExplanation.getValue(),
                            "weight(" + getQuery() + " in " + doc + ") ["
                                    + similarity.getClass().getSimpleName() + "], result of:",
                            scoreExplanation);
                }
            }
            return Explanation.noMatch("no matching term");
        }

        @Override
        public Scorer scorer(LeafReaderContext context) throws IOException {
            Similarity.SimScorer simScorer = similarity.simScorer(simWeight, context);
            List<Scorer> subScorers = new ArrayList<>();
            for (int i = 0; i < terms.length; i++) {
                TermState state = termContexts[i].get(context.ord);
                if (state != null) {
                    TermsEnum termsEnum = context.reader().terms(terms[i].field()).iterator();
                    termsEnum.seekExact(terms[i].bytes(), state);
                    PostingsEnum postings = termsEnum.postings(null, PostingsEnum.FREQS);
                    subScorers.add(new TermScorerWrapper(this, postings, simScorer, factors[i], terms[i]));
                }
            }
            if (subScorers.isEmpty()) return null;
            if (subScorers.size() == 1) return subScorers.get(0);
            return new SynonymScorer(simScorer, this, subScorers);
        }

        @Override
        public boolean isCacheable(LeafReaderContext ctx) {
            return true;
        }

        public TermContext build(IndexReaderContext context, Term term, double TFWeight) throws IOException {
            assert context != null && context.isTopLevel;
            final String field = term.field();
            final BytesRef bytes = term.bytes();
            final TermContext perReaderTermState = new TermContext(context);
            for (final LeafReaderContext ctx : context.leaves()) {
                final Terms terms = ctx.reader().terms(field);
                if (terms != null) {
                    final TermsEnum termsEnum = terms.iterator();
                    if (termsEnum.seekExact(bytes)) {
                        final TermState termState = termsEnum.termState();
                        long TF = termsEnum.totalTermFreq();
                        long newTF = Math.round(TFWeight * TF);
                        if (newTF <= termsEnum.docFreq()) newTF = termsEnum.docFreq() + 1;
                        perReaderTermState.register(termState, ctx.ord, termsEnum.docFreq(), newTF);
                    }
                }
            }
            return perReaderTermState;
        }
    }

    static class SynonymScorer extends DisjunctionScorer {
        private final Similarity.SimScorer similarity;

        SynonymScorer(Similarity.SimScorer similarity, Weight weight, List<Scorer> subScorers) {
            super(weight, subScorers, true);
            this.similarity = similarity;
        }

        @Override
        protected float score(DisiWrapper topList) throws IOException {
            float freq = tf(topList);
            if (freq == 0) return 0;
            return similarity.score(topList.doc, freq);
        }

        /**
         * THE ONLY CHANGE: the weighted frequencies are summed as FRACTIONS
         * first and rounded ONCE at the end. Low-weight variants can therefore
         * build up evidence together, while tf still stays an integer.
         */
        final int tf(DisiWrapper topList) throws IOException {
            double sum = 0.0;
            for (DisiWrapper w = topList; w != null; w = w.next) {
                TermScorerWrapper scorerWrapper = (TermScorerWrapper) w.scorer;
                sum += scorerWrapper.getScorer().freq() * scorerWrapper.getFactor();
            }
            long r = Math.round(sum);
            if (r < 0) r = 0;                       // negatif faktor gelirse guvenlik
            return (int) Math.min(r, Integer.MAX_VALUE);
        }
    }

    static class TermScorerWrapper extends Scorer {
        private TermScorer scorer;
        private double factor;
        private Term term;

        public TermScorerWrapper(SynonymWeight synonymWeight, PostingsEnum postings,
                                 SimScorer simScorer, double factor, Term term) {
            super(synonymWeight);
            this.scorer = new TermScorer(synonymWeight, postings, simScorer);
            this.factor = factor;
            this.term = term;
        }

        public TermScorer getScorer() { return scorer; }
        public double getFactor() { return factor; }

        @Override public int docID() { return scorer.docID(); }
        @Override public float score() throws IOException { return scorer.score(); }
        @Override public DocIdSetIterator iterator() { return scorer.iterator(); }
    }
}
