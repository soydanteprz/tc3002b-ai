import java.util.*;
import java.io.*;

public class C1627 {
    public static void main(String[] args) throws IOException {
        Scanner sc = new Scanner(System.in);
        PrintWriter pw = new PrintWriter(System.out);
        int t = sc.nextInt();
        while (t-- > 0) {
            int n = sc.nextInt();
            int[][] edges = new int[n - 1][2];
            int[] deg = new int[n];
            boolean valid = true;
            ArrayList<Integer>[] adjList = new ArrayList[n];
            for (int i = 0; i < n; i++) {
                adjList[i] = new ArrayList<>();
            }
            for (int i = 0; i < n - 1; i++) {
                int u = sc.nextInt() - 1;
                int v = sc.nextInt() - 1;
                edges[i] = new int[]{u, v};
                deg[u]++;
                deg[v]++;
                valid &= deg[u] <= 2 && deg[v] <= 2;
                adjList[u].add(i);
                adjList[v].add(i);
            }
            if (!valid) {
                pw.println(-1);
                continue;
            }
            int root = -1;
            for (int i = 0; i < n; i++) {
                if (adjList[i].size() == 1)
                    root = i;
            }
            int[] ans = new int[n - 1];
            int curColor = 2;
            int par = -1;
            while (true) {
                int nxt = -1;
                for (int e : adjList[root]) {
                    int other = edges[e][0] ^ edges[e][1] ^ root;
                    if (other != par) {
                        ans[e] = curColor;
                        curColor = 5 - curColor;
                        nxt = other;
                        break;
                    }
                }
                if (nxt == -1)
                    break;
                par = root;
                root = nxt;
            }
            for (int x : ans) {
                pw.print(x + " ");
            }
            pw.println();
        }
        pw.close();
    }

    
}
