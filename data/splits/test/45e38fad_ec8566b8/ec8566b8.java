import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;
import java.util.concurrent.ThreadLocalRandom;

public class A {
    private static void sport(List<Integer>[] g, Map<W, Integer> map) {
        int n = g.length;
        for (int i = 0; i < n; i++) {
            if (g[i].size() > 2) {
                System.out.println(-1);
                return;
            }
        }
        int[] ans = new int[n - 1];
        //dfs(new C(-1, 0), g, ans, 3, new HashSet<>());
        Queue<int[]> queue = new LinkedList<>();
        Set<Integer> seen = new HashSet<>();
        int val = 3;
        for (Integer integer : g[0]) {
            Integer idx = map.get(new W(0, integer));
            ans[idx] = val;
            queue.add(new int[]{val, integer});
            seen.add(integer);
            val = val == 2 ? 3 : 2;
        }
        seen.add(0);
        while (!queue.isEmpty()) {
            int[] poll = queue.poll();
            for (Integer u : g[poll[1]]) {
                if (!seen.contains(u)) {
                    seen.add(u);
                    int curr = poll[0] == 2 ? 3 : 2;
                    Integer integer = map.get(new W(poll[1], u));
                    ans[integer] = curr;
                    queue.add(new int[]{curr, u});
                }
            }
        }
        for (int an : ans) {
            System.out.print(an + " ");
        }
        System.out.println();
    }

    static class W {
        int u;
        int v;

        public W(int u, int v) {
            this.u = u;
            this.v = v;
        }

        

        
    }

    

    

    public static void main(String[] args) throws IOException {
        FastScanner sc = new FastScanner();
        int t = sc.nextInt();
        for (int i = 0; i < t; i++) {
            int n = sc.nextInt();
            List<Integer>[] g = new ArrayList[n];
            for (int j = 0; j < n; j++) {
                g[j] = new ArrayList<>();
            }
            Map<W, Integer> map = new HashMap<>();
            for (int j = 0; j < n - 1; j++) {
                int u = sc.nextInt() - 1;
                int v = sc.nextInt() - 1;
                g[u].add(v);
                g[v].add(u);
                map.put(new W(u, v), j);
                map.put(new W(v, u), j);
            }
            sport(g, map);
        }
    }

    

    
}
