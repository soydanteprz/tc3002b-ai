import static java.lang.Math.min;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Scanner;

public class D_2 {
	public static void main(String[] args) throws Exception {
		new D_2().go();
	}

	int USE_FILE = 0;
	String INPUT = "5\r\n" +
			"1 3 1 4 5\r\n" +
			"";

	void solve(int caseNr) {
		int n = in.nextInt();
		int[] h = na(n);
		int[] dp = new int[n];

		for (int i = 0; i < n; i++) {
			dp[i] = inf;
		}
		dp[0] = 0;


		// vector<pair<int, int>> st;
		LinkedList<int[]> st = new LinkedList<>();

		// lge[maxn], lle[maxn], rge[maxn], rle[maxn];
		int[] lge = new int[n];
		int[] lle = new int[n];
		int[] rge = new int[n];
		int[] rle = new int[n];


		for (int i = 0; i < n; i++) { // the nearest greater from the left
			while (!st.isEmpty() && st.getLast()[0] < h[i]) {
				st.removeLast();
			}
			if (st.isEmpty()) {
				lge[i] = -1;
			} else {
				lge[i] = st.getLast()[1];
			}
			st.add(new int[]{h[i], i});
		}

		st.clear();
		for (int i = 0; i < n; i++) { // the nearest less from the left
			while (!st.isEmpty() && st.getLast()[0] > h[i]) {
				st.removeLast();
			}
			if (st.isEmpty()) {
				lle[i] = -1;
			} else {
				lle[i] = st.getLast()[1];
			}
			st.add(new int[]{h[i], i});
		}

		st.clear();
		for (int i = n - 1; i >= 0; i--) { // the nearest greater from the right
			while (!st.isEmpty() && st.getLast()[0] < h[i]) {
				st.removeLast();
			}
			if (st.isEmpty()) {
				rge[i] = -1;
			} else {
				rge[i] = st.getLast()[1];
			}
			st.add(new int[]{h[i], i});
		}

		st.clear();
		for (int i = n - 1; i >= 0; i--) { // the nearest less from the right
			while (!st.isEmpty() && st.getLast()[0] > h[i]) {
				st.removeLast();
			}
			if (st.isEmpty()) {
				rle[i] = -1;
			} else {
				rle[i] = st.getLast()[1];
			}
			st.add(new int[]{h[i], i});
		}


		List<List<Integer>> jumps = new ArrayList<>();
		for (int i = 0; i < n; i++) {
			jumps.add(new ArrayList<>());
		}

		for (int i = 0; i < n; i++) {
			if (rle[i] != -1) jumps.get(i).add(rle[i]);
			if (rge[i] != -1) jumps.get(i).add(rge[i]);
			if (lle[i] != -1) jumps.get(lle[i]).add(i);
			if (lge[i] != -1) jumps.get(lge[i]).add(i);
		}
		for (int i = 0; i < n; i++) {
			for (int to : jumps.get(i)) {
				dp[to] = min(dp[to], dp[i] + 1);
			}
		}


		// System.out.println(jumps);


		System.out.println(dp[n - 1]);


		return;
	} // end of solve2





	// ---------------------------------------------------------------------------------------------- //

	void testCases() {
		int t = 1;
		for (int i=1; i<=t; i++) {
			solve(i);
		}
		output_writer.flush();
	}




	// ---------------------------------------------------------------------------------------------- //

	private int[] na(int n) {
		int[] a = new int[n];
		for (int i=0; i<n; i++) {
			a[i] = in.nextInt();
		}
		return a;
	}

	private int inf = Integer.MAX_VALUE;




	// ---------------------------------------------------------------------------------------------- //
	// ---------------------------------------------------------------------------------------------- //

	static boolean LOCAL = java.lang.System.getProperty("ONLINE_JUDGE") == null;
	Scanner in;
	PrintWriter output_writer = new PrintWriter(java.lang.System.out);

	// the output musn't flush automatic or we time out
	static class System {
		static PrintWriter out = null;
	}

	void go() throws Exception {
		System.out = output_writer;

		if (LOCAL && USE_FILE == 0) {
			//  if not set expects input from terminal
			java.lang.System.setIn(new ByteArrayInputStream(INPUT.getBytes()));
			in = new Scanner(java.lang.System.in);
		}
		if (LOCAL && USE_FILE > 0) {
			String filename = String.format("in/cf%d.txt", USE_FILE);
			File inputFile = new File(filename);
			in = new Scanner(inputFile);
			output_writer.printf("[%s]\n", filename);
			output_writer.flush();
		}
		if (!LOCAL) {
			in = new Scanner(java.lang.System.in);
		}


		long startTime = java.lang.System.currentTimeMillis();
		testCases();
		if (LOCAL) {
			java.lang.System.out.printf("[%dms]\n", java.lang.System.currentTimeMillis()-startTime);
		}
		in.close();
	}
}
