import java.util.*;
import java.io.*;

public class D {
    static ArrayList<Integer> adj[] = new ArrayList[(int)1e5+7];
    static int diameter = 0;
    static int[] depth = new int[(int)1e5 + 7];
    public static void main(String[] args) {
        FastReader in = new FastReader();
        int t = in.nextInt();
        while(t-- > 0){
            int n = in.nextInt();
            int a = in.nextInt(), b = in.nextInt(), da = in.nextInt(), db = in.nextInt();
            for(int i = 1; i <= n; i++){
                adj[i] = new ArrayList<>();
            }
            for(int i = 1; i <=n; i++){
                adj[i].clear();
            }
            for(int i = 0; i < n- 1; i++){
                int u = in.nextInt();
                int v = in.nextInt();
                adj[u].add(v);
                adj[v].add(u);
            }
            diameter = 0;
            depth[a] = 0;
            dfs(a, -1);
            System.out.println(2 * da >= Math.min(diameter, db) || depth[b] <= da ? "Alice" : "Bob");
        }
    }
    static int dfs(int node, int parent){
        int len = 0;
        for(int x : adj[node]){
            if(x != parent){
                depth[x] = depth[node] + 1;
                int cur = 1 + dfs(x, node);
                diameter = Math.max(diameter, cur + len);
                len = Math.max(len, cur);
//                System.out.print("x " + x + " node " + node + " par " + parent);
//                System.out.println(" cur " + cur + " len " + len + " diam " + diameter);
            }
        }
        return len;
    }
    

    
}
