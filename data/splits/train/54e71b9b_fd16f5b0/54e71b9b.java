import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.InputMismatchException;

public class C {
	// PRIMARY VARIABLES
	private static int n, m, k, x, y;
	private static int[][] a, b;
	private static long ans;
	private static int[][] id;
	private static long[][] dp;
	private static int idf;
	private static String s, t;
	private static HashMap<Integer, ArrayList<Integer>> g;

	// CONSTANTS
	private static final int MOD = (int) 1e9 + 7;
	private static final int[] dx = { -1, 1, 0, 0 };
	private static final int[] dy = { 0, 0, -1, 1 };
	private static final int MAX = Integer.MAX_VALUE;
	private static final int MIN = Integer.MIN_VALUE;
	private static final long MAXLONG = (long) 1e18;
	private static final long MINLONG = -(long) 1e18;

	public static void main(String[] args) {
		int testCases = in.nextInt();

		id = new int[(int)1e5+10][2];
		dp = new long[(int)1e5+10][2];
		
		for (int z = 1; z <= testCases; z++) {
			initCase(z);

			n = ini();
			a = ina2d(n, 2);
			
			g = intree(n);
			
			println(Math.max(dfs(0, -1, 0), dfs(0, -1, 1)));
		}

		//Ax,  A*(B-x)

		out.flush();
		out.close();

	}
	
	private static long dfs(int u, int p, int f) {
		if (id[u][f]==idf) {
			return dp[u][f];
		}
		
		
		long ans = 0;
		
		for(int v: g.get(u)) {
			if (v==p) continue;
			
			ans += Math.max(Math.abs(a[v][1]-a[u][f])+dfs(v, u, 1), Math.abs(a[v][0]-a[u][f])+dfs(v, u, 0));
		}
		id[u][f] = idf;
		dp[u][f] = ans;
		
		return ans;
	}

	// INIT
	private static void initCase(int z) {
		idf = z;
		ans = 0;
	}

	// PRINT ANSWER
	

	

	

	// SORT SHORTCUTS - QUICK SORT TO MERGE SORT
	

	

	// INPUT SHORTCUTS

	private static int[] ina(int n) {
		int[] temp = new int[n];
		for (int i = 0; i < n; i++) {
			temp[i] = in.nextInt();
		}
		return temp;
	}

	private static int[][] ina2d(int n, int m) {
		int[][] temp = new int[n][m];
		for (int i = 0; i < n; i++) {
			temp[i] = ina(m);
		}
		return temp;
	}

	

	private static int ini() {
		return in.nextInt();
	}

	

	

	

	// PRINT SHORTCUTS
	private static void println(Object... o) {
		for (Object x : o) {
			out.write(x + "");
		}
		out.write("\n");
	}

	

	

	// GRAPH SHORTCUTS
	private static HashMap<Integer, ArrayList<Integer>> intree(int n) {

		HashMap<Integer, ArrayList<Integer>> g = new HashMap<>();
		for (int i = 0; i < n; i++) {
			g.put(i, new ArrayList<>());
		}

		for (int i = 0; i < n - 1; i++) {
			int u = ini() - 1;
			int v = ini() - 1;
			g.get(u).add(v);
			g.get(v).add(u);
		}

		return g;
	}

	

	

	

	

	

	

	

	// MATHS AND NUMBER THEORY SHORTCUTS
	

	

	

	// Segment Tree
	

	// DSU
	

	// KMP
	

	

	

	

	

	// LCA
	

	// Geometry
	

	// FAST INPUT OUTPUT LIBRARY
	private static InputReader in = new InputReader(System.in);
	private static PrintWriter out = new PrintWriter(System.out);

	
}
