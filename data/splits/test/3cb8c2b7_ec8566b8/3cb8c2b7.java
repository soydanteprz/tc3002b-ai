import java.util.*;

public class CvsMe {
    private static class pair {
        int low, high;

        pair(int low, int high) {
            this.low = low;
            this.high = high;
        }
        

        // Equal objects must produce the same
        // hash code as long as they are equal
        
    }



    static int val[];
    static HashMap<pair, Integer> mp;
    static int ans[];

    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        int t = in.nextInt();
        while (t-- != 0) {
            int n = in.nextInt();
            ArrayList<ArrayList<Integer>> edge = new ArrayList<>();
            boolean flag = false;
            int deg[] = new int[n + 1];
            val = new int[n + 1];
            ans = new int[n + 1];
            mp = new HashMap<>();
            for (int i = 0; i <= n; i++) {
                edge.add(new ArrayList<Integer>());
            }
            int start = -1;
            int end = -1;
            for (int i = 1; i < n; i++) {
                int u = in.nextInt();
                int v = in.nextInt();
                pair p = new pair(u, v);
                mp.put(p, i);
                edge.get(u).add(v);
                edge.get(v).add(u);
                deg[u]++;
                deg[v]++;
                if (deg[u] > 2 || deg[v] > 2) {
                    flag = true;
                }

            }
            boolean vis[] = new boolean[n + 1];
            if (flag) {
                System.out.println(-1);
                continue;
            } else {
                for (int i = 1; i <= n; i++) {
                    if (deg[i] == 1 ) {
                        start = i;break;
                    }
                }

                dfs(vis, edge, 5, start);
                for (int i = 1; i < n; i++) {
                    System.out.print(ans[i] + " ");
                }

                System.out.println();
            }

        }
    }

    private static void dfs(boolean vis[], ArrayList<ArrayList<Integer>> edge, int k, int i) {
        vis[i] = true;


        for (int j : edge.get(i)) {
            if (!vis[j]) {
                int w = 2;
                if (val[i] == 2 || val[j] == 2){ w = 5;}
                int id = 0;
                pair p = new pair(i, j);

                if (mp.containsKey(p)) {
                    id = mp.get(p);
                    
                } else {
                    p = new pair(j, i);
                    if (mp.containsKey(p)) {id = mp.get(p);

                }}
                ans[id] = w;

                val[i] = w;
                val[j] = w;
                dfs(vis, edge, k, j);
            }
        }

    }

}
