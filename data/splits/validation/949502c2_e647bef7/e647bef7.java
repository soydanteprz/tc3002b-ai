import java.io.*;
import java.util.*;

public class Main {
    private static final boolean N_CASE = true;
    private List<List<Integer>> g;
    private int a;
    private int b;
    private int da;
    private int db;
    private int max;
    private int ab;

    private int dfs(int u, int fa, int depth) {
        if (u == a) {
            ab = depth;
        }

        int m1 = 0, m2 = 0;
        for (int v : g.get(u)) {
            if (v != fa) {
                int m = dfs(v, u, depth + 1) + 1;
                if (m > m1) { m2 = m1; m1 = m; }
                else if (m > m2) { m2 = m; }
            }
        }
        max = Math.max(max, m1 + m2);
        int cmax = Math.max(m1, m2);
        max = Math.max(max, cmax + depth);
        return cmax;
    }

    private void solve() {
        int n = sc.nextInt();
        a = sc.nextInt() - 1; b = sc.nextInt() - 1;
        da = sc.nextInt(); db = sc.nextInt();

        g = createGraph(n);
        for (int i = 0; i < n - 1; ++i) {
            int u = sc.nextInt() - 1, v = sc.nextInt() - 1;
            g.get(u).add(v);
            g.get(v).add(u);
        }
        max = 0;
        dfs(b, -1, 0);
        db = Math.min(max, db);

        boolean win = true;
        if (ab > da) {
            if (db > da * 2) {
                win = false;
            }
        }

        out.println(win ? "Alice" : "Bob");
    }

    private void run() {
        int T = N_CASE ? sc.nextInt() : 1;
        for (int t = 0; t < T; ++t) {
            solve();
        }
    }

    private static MyWriter out;
    private static MyScanner sc;

    

    private static class MyWriter extends PrintWriter {
        private MyWriter(OutputStream outputStream) {
            super(outputStream);
        }

        

        

        void println(int[] a) {
            for (int v : a) {
                println(v);
            }
        }

        

        void println(List<Integer> list) {
            list.forEach(this::println);
        }
    }

    private <T> List<List<T>> createGraph(int n) {
        List<List<T>> g = new ArrayList<>();
        for (int i = 0; i < n; ++i) {
            g.add(new ArrayList<>());
        }
        return g;
    }

    

    

    public static void main(String[] args) {
        out = new MyWriter(new BufferedOutputStream(System.out));
        sc = new MyScanner();
        new Main().run();
        out.close();
    }
}
