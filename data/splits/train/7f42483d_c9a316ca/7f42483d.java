import java.io.*;
import java.util.*;

public class Main
{
	static final boolean INPUT_FROM_FILE = false;
	static final String INPUT_FILE = "input/input.txt";
	static final String OUTPUT_FILE = "input/output.txt";
	static final long M = (long) 1e9 + 7;
	static FastReader in;
	static FastWriter out;

	static
	{
		try
		{
			in = new FastReader();
			out = new FastWriter();
		}
		catch (Exception e)
		{
			e.printStackTrace();
		}
	}

	public static void main(String[] args) throws IOException
	{
		int t = in.nextInt();
		while (t-- > 0) solve();
		out.close();
	}

	static long[][] dp;

	static long dfs(List<List<Integer>> tree, int parent, int current, int parentState, int[][] vRange) {
		if(dp[current][parentState] != -1) {
			return dp[current][parentState];
		}
		long left = Math.abs(vRange[current][0] - vRange[parent][parentState]);
		long right = Math.abs(vRange[current][1] - vRange[parent][parentState]);
		for(int child : tree.get(current)) {
			if(child != parent) {
				left += dfs(tree, current, child, 0, vRange);
				right += dfs(tree, current, child, 1, vRange);
			}
		}
		dp[current][parentState] = Math.max(left, right);
		return dp[current][parentState];
	}


	private static void solve()
	{
		int n = in.nextInt();
		int[][] vRange = new int[n+1][2];
		for(int i=1; i<=n; i++) {
			int l = in.nextInt(), r = in.nextInt();
			vRange[i][0] = l;
			vRange[i][1] = r;
		}
		List<List<Integer>> tree = new ArrayList<>();
		for(int i=0; i<=n; i++) tree.add(new LinkedList<>());
		for(int i=0; i<n-1; i++) {
			int u = in.nextInt();
			int v = in.nextInt();
			tree.get(u).add(v);
			tree.get(v).add(u);
		}

		dp = new long[n+1][2];
		for(int i=0; i<=n; i++) {
			Arrays.fill(dp[i], -1);
		}
		long left = 0, right = 0;

		for(int v : tree.get(1)) {
			left += dfs(tree, 1, v, 0, vRange);
			right += dfs(tree, 1, v, 1, vRange);
		}

		out.println(Math.max(left, right));

	}

	

	
}
