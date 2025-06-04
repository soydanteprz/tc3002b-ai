import java.io.OutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.util.StringTokenizer;
import java.io.IOException;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.io.InputStream;


public class Main {
    public static void main(String[] args) {
        InputStream inputStream = System.in;
        OutputStream outputStream = System.out;
        InputReader in = new InputReader(inputStream);
        PrintWriter out = new PrintWriter(outputStream);
        DTreeTag solver = new DTreeTag();
        int testCount = Integer.parseInt(in.next());
        for (int i = 1; i <= testCount; i++)
            solver.solve(i, in, out);
        out.close();
    }

    static class DTreeTag {
        int diam = 0;

        public int dfs(ArrayList<Integer> g[], int x, int depth[], int p) {
            int len = 0;
            for (int y : g[x]) {
                if (y != p) {
                    depth[y] = depth[x] + 1;
                    int cur = 1 + dfs(g, y, depth, x);
                    diam = Math.max(diam, cur + len);
                    len = Math.max(len, cur);
                }
            }
            return len;
        }

        public void solve(int testNumber, InputReader in, PrintWriter out) {
            int n = in.nextInt();
            int a = in.nextInt() - 1;
            int b = in.nextInt() - 1;
            int da = in.nextInt();
            int db = in.nextInt();
            int dis[] = new int[n];
            ArrayList<Integer> g[] = new ArrayList[n];
            for (int i = 0; i < n; i++) g[i] = new ArrayList<>();
            for (int i = 0; i < n - 1; i++) {
                int u = in.nextInt() - 1;
                int v = in.nextInt() - 1;
                g[u].add(v);
                g[v].add(u);
            }
            diam = 0;
            dfs(g, a, dis, -1);
            int disb = dis[b];
            if (2 * da >= Math.min(diam, db) || disb <= da) {
                out.println("Alice");
            } else {
                out.println("Bob");
            }

        }

    }

    
}
