import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.Queue;
import java.util.StringTokenizer;

public class Learning {
    static LinkedList<Integer>[] adj;

    public static void main(String[] args) throws Exception {
        FastInput in = new FastInput();
        StringBuilder sb = new StringBuilder();
        int t = in.nextInt();
        while (t-- > 0) {
            int n = in.nextInt();
            int a = in.nextInt() - 1;
            int b = in.nextInt() - 1;
            int da = in.nextInt();
            int db = in.nextInt();
            adj = new LinkedList[n];
            for (int i = 0; i < n; i++) {
                adj[i] = new LinkedList<>();
            }
            for (int i = 0; i < n - 1; i++) {
                int u = in.nextInt() - 1;
                int v = in.nextInt() - 1;
                adj[u].add(v);
                adj[v].add(u);
            }
            boolean f = solve(n, a, b, da, db);
            if (f) {
                sb.append("Bob");
            } else {
                sb.append("Alice");

            }
            sb.append("\n");
        }
        System.out.println(sb.toString());
    }

    private static boolean solve(int n, int a, int b, int da, int db) {
        if (db <= da * 2) {
            return false;
        }
        Queue<Integer> que = new LinkedList<>();
        que.add(a);
        int[] dist = new int[n];
        Arrays.fill(dist, -1);
        dist[a] = 0;
        while (!que.isEmpty()) {
            int t = que.poll();
            for (int i : adj[t]) {
                if (dist[i] == -1) {
                    que.add(i);
                    dist[i] = dist[t] + 1;
                }
            }
        }
        if (dist[b] <= da) {
            return false;
        }
        int maxPath = 0;
        for (int i = 0; i < n; i++) {
            maxPath = Math.max(maxPath, dfs(i, new boolean[n]));
        }
        if (2 * da >= maxPath-1) {
            return false;
        }
        return true;
    }

    private static int dfs(int i, boolean[] vis) {
        vis[i] = true;
        int max = 0;
        for (int v : adj[i]) {
            if (!vis[v]) {
                max = Math.max(dfs(v, vis), max);
            }
        }
        return max + 1;
    }

    
}



