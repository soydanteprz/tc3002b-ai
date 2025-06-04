import java.util.*;
import java.io.*;
import java.math.*;

/**
 *
 * @Har_Har_Mahadev
 */

/**
 * Main , Solution , Remove Public
 */
public class A {

	private static long[][] dp;
	private static ArrayList<Integer> lis1,lis0;

	public static void process() throws IOException {

		int n = sc.nextInt();
		int arr[] = sc.readArray(n);
		lis1 = new ArrayList<Integer>();
		lis0 = new ArrayList<Integer>();
		for(int i = 0; i<n; i++) {
			if(arr[i] == 1)lis1.add(i+1);
			else lis0.add(i+1);
		}
		Collections.sort(lis0);
		Collections.sort(lis1);
		int n0 = lis0.size();
		int n1 = lis1.size();
		dp = new long[n0+1][n1+1];
		for(int i = 0; i<=n0; i++) {
			Arrays.fill(dp[i], -1);
		}
		long ans = solve(0,0,n0,n1);
		System.out.println(ans);
		

	}

	private static long solve(int i, int j, int n0, int n1) {
		if(j == n1)return 0;
		if(i == n0)return INF;
		if(dp[i][j] != -1)return dp[i][j];
		long ans = solve(i+1, j, n0, n1);
		ans = Math.min(ans,Math.abs(lis0.get(i) - lis1.get(j)) + solve(i+1, j+1, n0, n1));
		return dp[i][j] = ans;
	}

	//=============================================================================
	//--------------------------The End---------------------------------
	//=============================================================================
	private static long INF = 2000000000000000000L;

	

	static FastScanner sc;
	static PrintWriter out;

	public static void main(String[] args) throws IOException {
		boolean oj = true;
		if (oj) {
			sc = new FastScanner();
			out = new PrintWriter(System.out);
		} else {
			sc = new FastScanner(100);
			out = new PrintWriter("output.txt");
		}
		int t = 1;
//		t = sc.nextInt();
		int TTT = 1;
		while (t-- > 0) {
			//			google(TTT++);
			process();
		}
		out.flush();
		out.close();
	}

	

	/////////////////////////////////////////////////////////////////////////////////////////////////////////


	

	

	

	

	


	

	

	


	

	

	

	

	

	

	/////////////////////////////////////////////////////////////////////////////////////////////////////////

	

	

	
}
