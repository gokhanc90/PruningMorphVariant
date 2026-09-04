import edu.anadolu.analysis.Analyzers;
import edu.anadolu.analysis.Tag;
import edu.anadolu.datasets.CollectionFactory;
import edu.anadolu.datasets.DataSet;
import edu.anadolu.qpp.Commonality;
import edu.anadolu.similarities.*;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.*;
import org.apache.lucene.search.similarities.ModelBase;
import org.apache.lucene.store.FSDirectory;
import org.clueweb09.InfoNeed;
import org.clueweb09.tracks.Track;

import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Runs the MLFN method of Singh &amp; Bhowmick (2022).
 *
 * This is the weighted-query path of Searcher.search(), with one difference:
 * MlfnWeightFunctions is used instead of SynonymWeightFunctions (predicted
 * NPMI). The same scoring is therefore applied without touching the source of
 * the retrieval toolkit.
 *
 * Usage:
 *   RunMlfn &lt;tfd_home&gt; &lt;collection&gt; &lt;tag&gt; &lt;synonymFile&gt; &lt;predictionFile&gt;
 *           &lt;ART|WMA&gt; &lt;runsPath&gt; &lt;models&gt;
 */
public class RunMlfn {

    static final String FIELD = "contents";
    static final String FIELD_ID = "id";

    static ModelBase model(String n) {
        switch (n) {
            case "BM25k1.2b0.75":      return new BM25c(1.2, 0.75);
            case "DPH":                return new DPH();
            case "DFIC":               return new DFIC();
            case "DFRee":              return new DFRee();
            case "DLH13":              return new DLH13();
            case "LGDc1.0":            return new LGDc(1.0);
            case "PL2c1.0":            return new PL2c(1.0);
            case "DirichletLMc2500.0": return new DirichletLM(2500.0);
            default: throw new IllegalArgumentException("unknown model: " + n);
        }
    }

    public static void main(String[] args) throws Exception {
        String tfd = args[0], coll = args[1], tagName = args[2];
        String synFile = args[3], predFile = args[4];
        boolean useWMA = "WMA".equalsIgnoreCase(args[5]);
        String runsPath = args[6];
        String[] modelNames = args[7].split(",");

        DataSet ds = CollectionFactory.dataset(
                edu.anadolu.datasets.Collection.valueOf(coll), tfd);
        Path indexPath = ds.indexesPath().resolve(Tag.NoStem.toString());
        Path synPath = Paths.get(ds.collectionPath().toString(), synFile);
        Path predPath = Paths.get(ds.collectionPath().toString(), predFile);
        Tag tag = Tag.valueOf(tagName);          // must contain "QBS" so that QBSART is called

        if (!tag.toString().contains("QBS"))
            throw new IllegalArgumentException(
                    "the tag must contain 'QBS' so SynonymWeightedQuery calls QBSART: " + tag);

        BooleanQuery.setMaxClauseCount(Integer.MAX_VALUE);
        IndexReader reader = DirectoryReader.open(FSDirectory.open(indexPath));
        System.out.println("index: " + indexPath + "  numDocs=" + reader.numDocs());

        // predFile == "NONE" -> full NPMI (from the index) = the original QBS
        // baseline. Same code path, only the source of NPMI differs, which is what
        // makes "how much does prediction cost?" a controlled comparison.
        SynonymWeightFunctions func;
        if ("NONE".equalsIgnoreCase(predFile)) {
            func = new SynonymWeightFunctions(ds, Tag.NoStem, tag);
            System.out.println("weighting: QBS (full NPMI, from the index)");
        } else {
            func = new MlfnWeightFunctions(ds, Tag.NoStem, tag, predPath, useWMA);
        }
        Commonality com = new Commonality(ds.indexesPath().resolve(Tag.NoStem.toString()));

        long t0 = System.currentTimeMillis();
        for (String mn : modelNames) {
            ModelBase sim = model(mn.trim());
            for (Track track : ds.tracks()) {
                Path dir = Paths.get(ds.collectionPath().toString(), runsPath,
                        tag.toString(), track.toString());
                Files.createDirectories(dir);
                String runTag = sim.toString().replaceAll(" ", "_")
                        + "_" + FIELD + "_" + tag + "_OR_all";

                IndexSearcher searcher = new IndexSearcher(reader);
                searcher.setSimilarity(sim);
                try (PrintWriter out = new PrintWriter(
                        Files.newBufferedWriter(dir.resolve(runTag + ".txt"),
                                StandardCharsets.US_ASCII))) {

                    for (InfoNeed need : track.getTopics()) {
                        String qs = need.query();
                        // NOTE: the analyzer(Path) overload expects a FILE path
                        // (parent + file name). analyzer(Tag, Path) expects a DIRECTORY
                        // and looks for <dir>/<Tag>.txt, which is wrong for synonym
                        // files with custom names.
                        Map<Integer, List<String>> syn =
                                Analyzers.getAnalyzedTokensWithSynonym(
                                        qs, Analyzers.analyzer(synPath));
                        List<String> orj = Analyzers.getAnalyzedTokens(
                                qs, Analyzers.analyzer(Tag.NoStem));

                        BooleanQuery.Builder b = new BooleanQuery.Builder();
                        int j = 0;
                        for (Map.Entry<Integer, List<String>> e : syn.entrySet()) {
                            if (j >= orj.size()) break;
                            String orjTerm = orj.remove(j);
                            Term[] otherOrj = orj.stream().map(o -> new Term(FIELD, o))
                                    .collect(Collectors.toList()).toArray(new Term[0]);
                            Term[] variants = e.getValue().stream()
                                    .map(t -> new Term(FIELD, t))
                                    .collect(Collectors.toList()).toArray(new Term[0]);
                            // SumRoundSynonymQuery: the weighted frequencies are first
                            // summed as fractions and rounded once at the very end. QBS
                            // and MLFN share this scheme, so the comparison stays
                            // controlled.
                            b.add(new BooleanClause(new SumRoundSynonymQuery(
                                    func, ds, otherOrj, tag, new Term(FIELD, orjTerm),
                                    com, variants), BooleanClause.Occur.SHOULD));
                            orj.add(j, orjTerm);
                            j++;
                        }
                        ScoreDoc[] hits = searcher.search(b.build(), 1000).scoreDocs;

                        if (hits.length == 0) {
                            out.println(need.id() + "\tQ0\t" + ds.getNoDocumentsID()
                                    + "\t1\t0\t" + runTag);
                            continue;
                        }
                        for (int i = 0; i < hits.length; i++) {
                            Document d = searcher.doc(hits[i].doc);
                            out.println(need.id() + "\tQ0\t" + d.get(FIELD_ID) + "\t"
                                    + (i + 1) + "\t" + hits[i].score + "\t" + runTag);
                        }
                    }
                }
                System.out.println("  written: " + track + " / " + mn);
            }
        }
        if (func instanceof MlfnWeightFunctions)
            System.out.println(((MlfnWeightFunctions) func).coverage());
        System.out.println("DONE (" + (System.currentTimeMillis() - t0) / 1000 + " s)");
        func.pmi.close();
        func.idf.close();
        reader.close();
    }

    /** SynonymKStemQBS -> SynonymKStem  (the base tag for the analyzer) */
    static String baseTag(Tag t) {
        return t.toString().replace("QBS", "");
    }
}
