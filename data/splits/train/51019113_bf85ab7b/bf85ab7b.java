import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;
import java.io.DataInputStream;
import java.io.FileInputStream;

import java.io.IOException;

public class ParsasHumongousTree {

    public static void main(String args[]) throws IOException {
        Reader scan = new Reader();
        StringBuilder sb = new StringBuilder();

        int t = scan.nextInt();
        while (t-- > 0) {
            int n = scan.nextInt();
            int[] l = new int[n + 1];
            int[] r = new int[n + 1];
            for (int i = 1; i <= n; i++) {
                l[i] = scan.nextInt();
                r[i] = scan.nextInt();
            }
            Graph g = new Graph(n);
            for (int i = 0; i < n - 1; i++) {
                g.addEdge(scan.nextInt(), scan.nextInt());
            }
            sb.append(g.dfs(l, r) + "\n");
        }
        System.out.println(sb);
    }
}

class Graph {

    ArrayList<Integer>[] node;
    int n;
    int c = 0;
    boolean[] vis;

    Graph(int s) {
        n = s + 1;
        vis = new boolean[n + 1];
        node = new ArrayList[n + 1];
        for (int i = 0; i < n + 1; i++) {
            node[i] = new ArrayList<>();
        }
    }

    void addEdge(int u, int v) {

        node[u].add(v);
        node[v].add(u);
        if (node[u].size() == 1) {
            c = u;
        }
        if (node[v].size() == 1) {
            c = v;
        }
    }

    void cleanVisArray() {
        for (int i = 0; i < n + 1; i++) {
            vis[i] = false;
        }
    }

    long dfs(int[] l, int[] r) {
        cleanVisArray();

        long[][] dp = new long[n][2];
        dfsMain(1, dp, l, r);
        return Math.max(dp[1][0], dp[1][1]);
    }

    void dfsMain(int v, long[][] dp, int[] l, int[] r) {
        vis[v] = true;
        for (int i : node[v]) {
            if (!vis[i]) {
                dfsMain(i, dp, l, r);
                dp[v][0] += Math.max(Math.abs(l[v] - l[i]) + dp[i][0], Math.abs(l[v] - r[i]) + dp[i][1]);
                dp[v][1] += Math.max(Math.abs(r[v] - l[i]) + dp[i][0], Math.abs(r[v] - r[i]) + dp[i][1]);
            }
        }
    }

}


