import java.io.OutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.util.InputMismatchException;
import java.io.IOException;
import java.util.ArrayList;
import java.io.InputStream;

public class Main {
    public static void main(String[] args) {
        InputStream inputStream = System.in;
        OutputStream outputStream = System.out;
        FastReader in = new FastReader(inputStream);
        PrintWriter out = new PrintWriter(outputStream);
        CParsasHumongousTree solver = new CParsasHumongousTree();
        int testCount = Integer.parseInt(in.next());
        for (int i = 1; i <= testCount; i++)
            solver.solve(i, in, out);
        out.close();
    }

    static class CParsasHumongousTree {
        int n;
        long[][] dp;
        int[] l;
        int[] r;
        ArrayList<Integer>[] a;

        public void solve(int testNumber, FastReader in, PrintWriter out) {
            n = in.nextInt();
            l = new int[n];
            r = new int[n];
            dp = new long[n][2];
            a = new ArrayList[n];
            for (int i = 0; i < n; ++i) {
                a[i] = new ArrayList<>();
            }

            for (int i = 0; i < n; ++i) {
                l[i] = in.nextInt();
                r[i] = in.nextInt();
            }

            for (int i = 0; i < n - 1; ++i) {
                int u = in.nextInt() - 1, v = in.nextInt() - 1;
                a[u].add(v);
                a[v].add(u);
            }

            dfs(0, -1);

            out.println(Math.max(dp[0][0], dp[0][1]));
        }

        void dfs(int u, int p) {
            dp[u][0] = dp[u][1] = 0;
            for (int v : a[u]) {
                if (v != p) {
                    dfs(v, u);
                    dp[u][0] += Math.max(dp[v][0] + Math.abs(l[u] - l[v]), dp[v][1] + Math.abs(l[u] - r[v]));
                    dp[u][1] += Math.max(dp[v][0] + Math.abs(r[u] - l[v]), dp[v][1] + Math.abs(r[u] - r[v]));
                }
            }
        }

    }

    
}
