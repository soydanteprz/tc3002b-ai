import java.io.*;
import java.util.*;

public class C {
    static InputReader in;
    static PrintWriter out;

    public static void main(String[] args) {
        //initReaderPrinter(true);
        initReaderPrinter(false);
        solve(in.nextInt());
        //solve(1);        
    }
    static int n;
    static List<Integer>[] adj;
    static long[] l, r;
    static void solve(int testCnt) {
        for (int testNumber = 0; testNumber < testCnt; testNumber++) {
            n = in.nextInt();
            adj = new List[n + 1];
            for(int i = 0; i <= n; i++) {
                adj[i] = new ArrayList<>();
            }
            l = new long[n + 1];
            r = new long[n + 1];
            for(int i = 1; i <= n; i++) {
                l[i] = in.nextInt();
                r[i] = in.nextInt();
            }
            for(int i = 0; i < n - 1; i++) {
                int u = in.nextInt();
                int v = in.nextInt();
                adj[u].add(v);
                adj[v].add(u);
            }
            long[] ans = dfs(1, 0);
            out.println(Math.max(ans[0], ans[1]));
        }
        out.close();
    }

    static long[] dfs(int curr, int par) {
        long[] ans = new long[2];
        for(int next : adj[curr]) {
            if(next == par) continue;
            long[] nextAns = dfs(next, curr);
            ans[0] += Math.max(nextAns[0] + Math.abs(l[curr] - l[next]), nextAns[1] + Math.abs(l[curr] - r[next]));
            ans[1] += Math.max(nextAns[0] + Math.abs(r[curr] - l[next]), nextAns[1] + Math.abs(r[curr] - r[next]));
        }
        return ans;
    }

    static void initReaderPrinter(boolean test) {
        if (test) {
            try {
                in = new InputReader(new FileInputStream("src/input.in"));
                out = new PrintWriter(new FileOutputStream("src/output.out"));
            } catch (IOException e) {
                e.printStackTrace();
            }
        } else {
            in = new InputReader(System.in);
            out = new PrintWriter(System.out);
        }
    }

    
}
