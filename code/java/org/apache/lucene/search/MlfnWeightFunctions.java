package org.apache.lucene.search;

import edu.anadolu.analysis.Tag;
import edu.anadolu.datasets.DataSet;
import org.apache.lucene.index.Term;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * The MLFN weighting of Singh &amp; Bhowmick (2022).
 *
 * It uses exactly the ART / WMA formulas of Paik et al. (2013); the only
 * difference is that NPMI is PREDICTED by MLFN rather than computed from the
 * index. The question "how much does prediction cost?" can therefore be answered
 * by comparing directly against the QBS baseline, which runs the same formulas.
 *
 * The class extends SynonymWeightFunctions and overrides the public QBSART, so
 * SynonymWeightedQuery can be used unmodified (it calls that method whenever the
 * tag contains "QBS").
 */
public class MlfnWeightFunctions extends SynonymWeightFunctions {

    /** "term\tvariant" -> predicted NPMI */
    private final Map<String, Double> pred = new HashMap<>();
    private final boolean useWMA;
    private int miss = 0, hit = 0;

    public MlfnWeightFunctions(DataSet dataset, Tag indexTag, Tag runningTag,
                               Path predictionFile, boolean useWMA) {
        super(dataset, indexTag, runningTag);
        this.useWMA = useWMA;
        try {
            List<String> lines = Files.readAllLines(predictionFile, StandardCharsets.UTF_8);
            for (String l : lines) {
                if (l.isEmpty() || l.startsWith("#")) continue;
                String[] p = l.split(",");
                if (p.length < 3) continue;
                pred.put(p[0] + "\t" + p[1], Double.valueOf(p[2]));
            }
        } catch (IOException e) {
            throw new RuntimeException("could not read the prediction file: " + predictionFile, e);
        }
        System.out.println("MLFN prediction table: " + pred.size() + " pairs  ("
                + (useWMA ? "WMA" : "ART") + ")");
    }

    /** Predicted NPMI. A pair absent from the table scores 0 (treated as unrelated). */
    private double npmiHat(Term a, Term b) {
        if (a.text().equals(b.text())) return 1.0;
        Double v = pred.get(a.text() + "\t" + b.text());
        if (v == null) v = pred.get(b.text() + "\t" + a.text());
        if (v == null) { miss++; return 0.0; }
        hit++;
        return v;
    }

    /** Paik et al. (2013): association strength, threshold 0.25 */
    private double assoc(double npmi) {
        return npmi >= 0.25 ? npmi : 0.0;
    }

    /** The replacement-strength component of ART: it compares how strongly the
     *  variant is associated with the OTHER query terms against how strongly the
     *  original term is associated with the same terms.
     *
     *  NOTE: this must behave exactly like SynonymWeightFunctions.RA(), otherwise
     *  MLFN and QBS would differ in a second way besides the source of NPMI.
     *  Original: sumOver==0 -> 0; otherwise sumOver/sumBelow (Infinity when
     *  sumBelow==0), and the caller applies Math.min(1.0, .) -> 1.0. */
    private double replacement(Term orj, Term v, Term[] otherOrj) throws IOException {
        double over = 0.0, below = 0.0;
        for (Term o : otherOrj) {
            over  += idf.value(o.text()) * assoc(npmiHat(v, o));
            below += idf.value(o.text()) * assoc(npmiHat(orj, o));
        }
        if (over == 0.0) return 0.0;
        return Math.min(1.0, over / below);   // below==0 -> Infinity -> 1.0 (same as the original)
    }

    /** WMA: the IDF-weighted mean association of the variant with every query term. */
    private double wma(Term v, Term orj, Term[] otherOrj) throws IOException {
        double num = idf.value(orj.text()) * assoc(npmiHat(v, orj));
        double den = idf.value(orj.text());
        for (Term o : otherOrj) {
            num += idf.value(o.text()) * assoc(npmiHat(v, o));
            den += idf.value(o.text());
        }
        return den == 0.0 ? 0.0 : num / den;
    }

    @Override
    public double QBSART(Term orj, Term v, Term[] otherOrj) {
        if (orj.equals(v)) return 1.0;
        try {
            if (useWMA) return wma(v, orj, otherOrj);
            return 0.7 * assoc(npmiHat(orj, v)) + 0.3 * replacement(orj, v, otherOrj);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public String coverage() {
        int t = hit + miss;
        return t == 0 ? "-" : String.format("prediction coverage: %d/%d (%.1f%%)",
                hit, t, 100.0 * hit / t);
    }
}
