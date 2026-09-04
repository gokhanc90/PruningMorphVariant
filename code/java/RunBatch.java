import edu.anadolu.Searcher;
import edu.anadolu.analysis.Tag;
import edu.anadolu.datasets.CollectionFactory;
import edu.anadolu.datasets.DataSet;
import edu.anadolu.similarities.*;
import org.apache.lucene.search.similarities.ModelBase;

import java.nio.file.*;
import java.util.*;

/**
 * For one synonym file, runs ALL tracks under the given models and writes the
 * results to TFD_HOME/&lt;collection&gt;/&lt;runsPath&gt;/&lt;Tag&gt;/&lt;track&gt;/.
 *
 * Usage:
 *   RunBatch &lt;tfd_home&gt; &lt;collection&gt; &lt;tag&gt; &lt;synonymFile&gt; &lt;runsPath&gt; &lt;models&gt; &lt;threads&gt;
 * Example:
 *   RunBatch C:\...\TFD_HOME CW09B SynonymKStem SynonymKStem_X_sbert.txt X_sbert BM25k1.2b0.75,DPH 4
 */
public class RunBatch {

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
        String tfd = args[0], coll = args[1], tagName = args[2], synFile = args[3];
        String runsPath = args[4];
        int threads = args.length > 6 ? Integer.parseInt(args[6]) : 4;

        Set<ModelBase> models = new LinkedHashSet<>();
        for (String m : args[5].split(",")) models.add(model(m.trim()));

        DataSet ds = CollectionFactory.dataset(
                edu.anadolu.datasets.Collection.valueOf(coll), tfd);
        Path indexPath = ds.indexesPath().resolve(Tag.NoStem.toString());

        System.out.println("== " + coll + " / " + synFile + " -> " + runsPath);
        long t0 = System.currentTimeMillis();

        // synFile == "NONE" -> no stemming, no synonym file is used
        Searcher s;
        if ("NONE".equalsIgnoreCase(synFile)) {
            s = new Searcher(indexPath, ds, 1000);
        } else {
            Path synPath = Paths.get(ds.collectionPath().toString(), synFile);
            if (!Files.exists(synPath))
                throw new IllegalArgumentException("synonym file not found: " + synPath);
            s = new Searcher(indexPath, ds, Tag.valueOf(tagName), synPath, 1000);
        }
        try (Searcher searcher = s) {
            searcher.searchWithThreads(threads, models,
                    Collections.singletonList("contents"), runsPath);
        }
        System.out.println("== DONE " + runsPath + "  (" +
                (System.currentTimeMillis() - t0) / 1000 + " s)");
    }
}
