import java.io.*;
import java.util.*;

public class Main {

//	static boolean[] prime = new boolean[10000000];
	final static long mod = 1000000007;

	public static void main(String[] args) {
//		sieve();
		InputReader in = new InputReader(System.in);
		PrintWriter out = new PrintWriter(System.out);

		int t = in.nextInt();
		while (t-- > 0) {
			int n = in.nextInt();
			int[][] a = new int[n - 1][2];
			ArrayList<ArrayList<Data>> g = new ArrayList<>();
			for (int i = 0; i < n - 1; i++) {
				g.add(new ArrayList<>());
				a[i][0] = in.nextInt() - 1;
				a[i][1] = in.nextInt() - 1;
			}
			g.add(new ArrayList<>());
			for (int i = 0; i < n - 1; i++) {
				g.get(a[i][0]).add(new Data(a[i][1], i));
				g.get(a[i][1]).add(new Data(a[i][0], i));
			}
			if (!check(g)) {
				System.out.println(-1);
				continue;
			}
			int[] ans = new int[n - 1];

			ans[0] = 2;
			for(int node: a[0]) {
				for(Data d: g.get(node)) {
					solve(g, d.a, d.ind, false, ans);
				}
			}

			for (int i : ans)
				System.out.print(i + " ");
			System.out.println();
		}

		out.flush();
	}

	private static void solve(ArrayList<ArrayList<Data>> g, int node, int edge, boolean b, int[] ans) {
		if (ans[edge] != 0)
			return;

		ans[edge] = b ? 2 : 3;
		for (Data d : g.get(node)) {
			solve(g, d.a, d.ind, !b, ans);
		}

	}

	private static boolean check(ArrayList<ArrayList<Data>> g) {
		for (ArrayList<Data> a : g)
			if (a.size() > 2)
				return false;
		return true;
	}

	

	

	

	

	

	

//	static void sieve() {
//		for (int i = 2; i * i < prime.length; i++) {
//			if (prime[i])
//				continue;	
//			for (int j = i * i; j < prime.length; j += i) {
//				prime[j] = true;
//			}
//		}
//	}

}

class Data {
	int a;
	int ind;

	Data(int val, int ind) {
		this.a = Math.abs(val);
		this.ind = ind;
	}
}




