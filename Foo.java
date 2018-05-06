import java.io.*;
import java.util.*;

class Foo {
    static String DATAFILE_PREFIX = "[main] testing potentials for data file: ";
    static String MARGINAL_PREFIX = "marginal probability distribution for node ";
    static String MAXLOG_PREFIX = "MaxLogProbability=";
    static String PX_PREFIX = "P(x_";
    static String X_PREFIX = "x_";
    static int M = 1000_000;
    
    static String wrap (String s, int code) {
        return (char) 27 + "[" + code + "m" + s + (char) 27 + "[0m";
    }

    public static void main (String[] args) {
        Scanner scan = new Scanner (System.in);

        Map<String, Integer> file2int = new HashMap<> ();
        String[] int2file = new String[20];
        int file_seed = 0;

        StringBuilder sb = new StringBuilder ();
        String line = "";

        outer: while (scan.hasNextLine ()) {
            if (line.length () == 0)
                line = scan.nextLine ();
            if (line.startsWith (DATAFILE_PREFIX)) {
                String datafile = line.substring (DATAFILE_PREFIX.length ());
                Map<Integer, Double> table = new HashMap<> ();
                String max_logp = null;
                Map<Integer, String> config = new HashMap<> ();
                int N = 0, K = 0;
                
                while (true) {
                    line = scan.nextLine ();
                    if (line.startsWith (MAXLOG_PREFIX)) {
                        String new_maxlogp = line.substring (MAXLOG_PREFIX.length ());
                        if (max_logp != null) {
                            if (!max_logp.equals (new_maxlogp)) {
                                System.out.printf ("Your max-sum product is producing inconsistent MaxLogProbability %s and %s, aborting...\n", wrap (max_logp, 91), wrap (new_maxlogp, 91));
                                System.exit (1);
                            }
                        } else {
                            max_logp = new_maxlogp;
                        }
                    } else if (!line.startsWith (DATAFILE_PREFIX)) {
                        if (!line.contains ("="))
                            continue;
                        line = line.replaceAll ("\\s+", "");
                        String[] tokens = line.split ("=");
                        if (tokens.length == 2) {
                            int bp_node = Integer.parseInt (tokens[0].substring (X_PREFIX.length ()));
                            String bp_val = tokens[1];
                            String old_val = config.get (bp_node);
                            if (old_val != null) {
                                if (!old_val.equals (bp_val)) {
                                    System.out.printf ("Your max-sum product is not producing consistent global configuration: phi(%s) has inconsistent values of %s and %s\nAborting...\n", wrap (bp_node + "", 91), wrap (old_val + "", 91), wrap (bp_val + "", 91));
                                    System.exit (1);
                                }
                            } else {
                                config.put (bp_node, bp_val);
                            }
                        } else {
                            int node = Integer.parseInt (tokens[0].substring (PX_PREFIX.length ()));
                            int k = Integer.parseInt (tokens[1].substring (0, tokens[1].length () - 1));
                            Double entry = Double.parseDouble (tokens[2]);
                            int key = node * M + k;
                            table.put (key, entry);
                            N = Math.max (N, node);
                            K = Math.max (K, k);
                        }
                    }
                    if (!scan.hasNextLine () || line.startsWith (DATAFILE_PREFIX)) {
                        sb.append (String.format ("Testing result for data file: %s\nmarginal probability distribution as NxK table:\n", wrap (datafile, 95)));
                        Double[][] mat = new Double[N + 1][K + 1];
                        for (Integer key : table.keySet ()) {
                            int node = key / M, k = key % M;
                            mat[node][k] = table.get (key);
                        }
                        sb.append (String.format ("%5s", ""));
                        for (int j = 1; j <= K; j++) {
                            sb.append (String.format ("%11d", j));
                        }
                        sb.append ("\n");
                        for (int i = 1; i <= N; i++) {
                            sb.append (String.format ("%14s", wrap (i + "", 96)));
                            for (int j = 1; j <= K; j++) {
                                sb.append (String.format ("   %7f", mat[i][j]));
                            }
                            sb.append ("\n");
                        }

                        sb.append ("MaxLogProbability=" + max_logp + "\n");
                        sb.append ("Best Configuration: \n");

                        for (int i = 1; i <= N; i++) {
                            sb.append (String.format ("%13s", wrap (i + "", 96)));
                        }
                        sb.append ("\n");
                        for (int i = 1; i <= N; i++) {
                            sb.append (String.format ("%4s", config.get (i)));
                        }
                        sb.append ("\n\n");
                        if (scan.hasNextLine ())
                            continue outer;
                        else
                            break outer;
                    }
                }
            }
        }
        System.out.println (sb.toString ());
    }
}