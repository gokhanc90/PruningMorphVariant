import edu.anadolu.analysis.Analyzers;
import edu.anadolu.analysis.Tag;
import edu.anadolu.datasets.CollectionFactory;
import edu.anadolu.datasets.DataSet;
import edu.anadolu.qpp.PMI;
import org.clueweb09.InfoNeed;
import org.clueweb09.tracks.Track;

import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

/**
 * Computes NPMI from the index ONCE for every (term, term) pair the ART
 * weighting needs, and writes the result out as a table.
 *
 * Why: SynonymWeightFunctions.calculateNPMI looks every pair up in the index, and
 * does so again for every query, every track and every model. That is why the QBS
 * runs took 9 hours. Once the table exists, MlfnWeightFunctions can read it and
 * the same result is obtained about 5 times faster.
 *
 * The pairs are enumerated on the Java side with EXACTLY the query construction of
 * RunMlfn, so that every term the analyzer produces appears in the table
 * (enumerating them in Python left coverage at 96.4%).
 *
 * NOTE: this is the NPMI of the WEIGHTING BASELINES (QBS / ART). The NPMI the
 * pruning method itself uses is a different quantity and comes from the workbooks
 * read by score_all.py -- see the README.
 *
 * Usage: DumpNpmi &lt;tfd&gt; &lt;collection&gt; &lt;tag&gt; &lt;synonymFile&gt; &lt;outputFile&gt;
 */
public class DumpNpmi {

    static final String FIELD = "contents";

    public static void main(String[] args) throws Exception {
        String tfd = args[0], coll = args[1], tagName = args[2], synFile = args[3];
        Path outFile = Paths.get(args[4]);

        DataSet ds = CollectionFactory.dataset(
                edu.anadolu.datasets.Collection.valueOf(coll), tfd);
        Path synPath = Paths.get(ds.collectionPath().toString(), synFile);
        Tag tag = Tag.valueOf(tagName);

        // --- 1) collect the pairs that are needed (same construction as RunMlfn)
        Set<String> pairs = new LinkedHashSet<>();
        int nq = 0;
        for (Track track : ds.tracks()) {
            for (InfoNeed need : track.getTopics()) {
                nq++;
                Map<Integer, List<String>> syn = Analyzers.getAnalyzedTokensWithSynonym(
                        need.query(), Analyzers.analyzer(synPath));
                List<String> orj = Analyzers.getAnalyzedTokens(
                        need.query(), Analyzers.analyzer(Tag.NoStem));
                int j = 0;
                for (Map.Entry<Integer, List<String>> e : syn.entrySet()) {
                    if (j >= orj.size()) break;
                    String orjTerm = orj.remove(j);
                    List<String> other = new ArrayList<>(orj);
                    for (String v : e.getValue()) {
                        if (v.equals(orjTerm)) continue;      // equal terms give NPMI=1, no entry needed
                        pairs.add(orjTerm + "\t" + v);        // assoc(NPMI(orj,v))
                        for (String o : other) pairs.add(v + "\t" + o);   // numerator of replacement
                    }
                    for (String o : other) pairs.add(orjTerm + "\t" + o); // denominator of replacement
                    orj.add(j, orjTerm);
                    j++;
                }
            }
        }
        System.out.println(coll + "/" + tagName + ": " + nq + " queries -> "
                + pairs.size() + " unique pairs");

        // --- 2) compute NPMI from the index
        PMI pmi = new PMI(ds.indexesPath().resolve(Tag.NoStem.toString()), FIELD);
        int done = 0, bad = 0;
        long t0 = System.currentTimeMillis();
        try (PrintWriter out = new PrintWriter(
                Files.newBufferedWriter(outFile, StandardCharsets.UTF_8))) {
            out.println("# term,term,npmi  -- FULL NPMI computed from the index (cache)");
            for (String p : pairs) {
                String[] ab = p.split("\t");
                double v;
                try {
                    v = pmi.npmi(ab[0], ab[1]);
                    if (Double.isNaN(v) || Double.isInfinite(v)) { v = 0.0; bad++; }
                } catch (Exception ex) {
                    v = 0.0; bad++;
                }
                out.printf(Locale.US, "%s,%s,%.6f%n", ab[0], ab[1], v);
                if (++done % 5000 == 0) {
                    long el = (System.currentTimeMillis() - t0) / 1000;
                    System.out.printf("   %d/%d  %d s  (eta %d s)%n",
                            done, pairs.size(), el, el * (pairs.size() - done) / done);
                }
            }
        }
        pmi.close();
        System.out.printf("DONE: %d pairs, %d undefined (written as 0), %d s -> %s%n",
                done, bad, (System.currentTimeMillis() - t0) / 1000, outFile);
    }
}
