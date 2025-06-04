import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.PriorityQueue;
import java.util.StringTokenizer;

public class Solve {

	static int mod = 1000000000 + 7;
	static long fact[] = new long[2 * 100000 + 3];

	public static void main(String[] args) throws Exception {
		Scanner sc = new Scanner();
		PrintWriter pw = new PrintWriter(System.out);
		int t = sc.nextInt();
		while (t-- > 0) {
			int n = sc.nextInt(); // blocks
			int m = sc.nextInt(); // towers to build
			int x = sc.nextInt(); // maximum diff
			int heights[] = new int[n];
			int sum = 0;
			PriorityQueue<Pair> pq = new PriorityQueue<Pair>();
			for (int i = 0; i < n; i++) {
				heights[i] = sc.nextInt();
			}
			for (int i = 0; i < m; i++) {
				pq.add(new Pair(i + 1, 0));
			}
			pw.println("YES");

			for (int i = 0; i < n; i++) {
				Pair p = pq.poll();
				p.value += heights[i];
				pq.add(p);
				pw.print(p.tower + " ");
			}

			pw.println();
		}
		pw.flush();
	}

	

	

	

	

}

class Pair implements Comparable<Pair> {
	int tower;
	int value = 0;

	Pair(int x, int y) {
		tower = x;
		value = y;
	}

	

}


