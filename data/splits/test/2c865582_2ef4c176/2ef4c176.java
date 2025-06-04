import java.io.*;
import java.util.*;

public class Template {

	static int mod = 1000000007;

	public static void main(String[] args) {
		FastScanner sc = new FastScanner();
		PrintWriter out = new PrintWriter(System.out);
		int yo = sc.nextInt();
		while (yo-- > 0) {
			int n = sc.nextInt();
			
			Map<Integer,Pair> map = new HashMap<>();
			
			for(int i = 0; i < n; i++) {
				map.put(i, new Pair(sc.nextLong(),sc.nextLong()));
			}
			
			List<List<Integer>> list = new ArrayList<>();
			for(int i = 0; i < n; i++) {
				list.add(new ArrayList<>());
			}
			for(int i = 0; i < n-1; i++) {
				int x = sc.nextInt()-1;
				int y = sc.nextInt()-1;
				list.get(x).add(y);
				list.get(y).add(x);
			}
			
			// l -> 0, r -> 1
			
			for(int i = 0; i < 1e5+3; i++) {
				for(int j = 0; j < 2; j++) {
					dp[i][j] = -1;
				}
			}
			long a1 = dfs(map,list,0,0,-1);
			long a2 = dfs(map,list,1,0,-1);
			long ans = Math.max(a1, a2);
			out.println(ans);
		}
		out.close();
	}

//	static Map<String,Long> dp = new HashMap<>();
	static long[][] dp = new long[(int)1e5+10][2];
	private static long dfs(Map<Integer, Pair> map, List<List<Integer>> list, 
							int x, int node, int parent) {
		
		
		if(dp[node][x] != -1) {
			return dp[node][x];
		}

		List<Integer> neighbours = list.get(node);
		
		long ans1 = 0;
		if(x == 0) {
			ans1 = map.get(node).x;
		}
		else {
			ans1 = map.get(node).y;
		}
		
		
		long uAns = 0;
		for(int e : neighbours) {
			if(e == parent) continue;
			long ua1 = dfs(map,list,0,e,node);
			long ua2 = dfs(map,list,1,e,node);
			// consider 0
			long a1 = ua1 + Math.abs(map.get(e).x-ans1);
			// consider 1
			long a2 = ua2 + Math.abs(map.get(e).y-ans1);
			uAns += Math.max(a1, a2);
		}
		
		return dp[node][x] = uAns;
	}

	static class Pair {
		long x;
		long y;

		public Pair(long x, long y) {
			this.x = x;
			this.y = y;
		}
	}

	

	

	

	

	

	//	For Input.txt and Output.txt	
	//	FileInputStream in = new FileInputStream("input.txt");
	//	FileOutputStream out = new FileOutputStream("output.txt");
	//	PrintWriter pw = new PrintWriter(out);
	//	Scanner sc = new Scanner(in);
}
