import java.util.*;
import java.io.*;
import java.math.*;
public class Euler {
	
	static int N = (int)1e5 + 5;
	static int n, a, b, da, db;
	static int[] depth = new int[N];
	static ArrayList<Integer>[] adj = new ArrayList[N];
	static int diam;
	
	public static int dfs(int x, int p) {
		
		int len = 0;
		for (int y : adj[x]) {
			if (y != p) {
				depth[y] = depth[x] + 1;
				int cur = 1 + dfs(y, x);
				diam = Math.max(diam, cur + len);
				len = Math.max(len, cur);
			}
		}
		return len;
		
	}
	
    public static void main(String[] args){

        FastReader in = new FastReader();
        PrintWriter o = new PrintWriter(System.out);
        
        int t = in.nextInt();
        
        while(t-- > 0) {
			n = in.nextInt();
			a = in.nextInt();
			b = in.nextInt();
			da = in.nextInt();
			db = in.nextInt();
			for (int i = 1; i <= n; i++) {
				adj[i] = new ArrayList<>();
			}
			for (int i = 0; i < n - 1; i++) {
				int u = in.nextInt();
				int v = in.nextInt();
				adj[u].add(v);
				adj[v].add(u);
			}
			diam = 0;
			depth[a] = 0;
			dfs(a, -1);
			
			boolean works = true;
			
			if (depth[b] <= da) {
				o.println("Alice");
				continue;
			}
			
			if (2 * da >= diam) {
				o.println("Alice");
				continue;
			}
			
			if (db > 2 * da) {
				o.println("Bob");
				continue;
			}
			
			if (db <= 2 * da) {
				o.println("Alice");
			}
			
			
		}
        
        
        o.close();
        o.flush();
        return;


    }
    
    
    


}
