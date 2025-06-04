import java.io.*;
import java.util.*;

public class A {
	static ArrayList<int[]>[] adj;
	static int[] vals = { 2, 11 };
	static int[] res;

	static void dfs(int u, int p, int par) {

		for (int[] nxt : adj[u]) {
			int v = nxt[0];
			int idx = nxt[1];
			if (v != p) {
				res[idx] = vals[par];
				dfs(v, u, 1 ^ par);
			}
		}

	}

	public static void main(String[] args) throws IOException {
		Scanner sc = new Scanner(System.in);
		PrintWriter pw = new PrintWriter(System.out);
		int t = sc.nextInt();
		while (t-- > 0) {
			int n = sc.nextInt();
			res = new int[n - 1];
			adj = new ArrayList[n];
			for (int i = 0; i < n; i++) {
				adj[i] = new ArrayList<int[]>();
			}
			for (int i = 0; i < n - 1; i++) {
				int u = sc.nextInt() - 1;
				int v = sc.nextInt() - 1;
				adj[u].add(new int[] { v, i });
				adj[v].add(new int[] { u, i });
			}

			boolean valid = true;
			int start = 0;
			for (int i = 0; i < n; i++) {
				if (adj[i].size() == 1)
					start = i;
				valid &= adj[i].size() <= 2;
			}

			dfs(start, -1, 0);
			if (valid) {
				for (int x : res)
					pw.print(x + " ");
			} else {
				pw.print(-1);
			}
			pw.println();
		}
		pw.flush();
	}

	

}
