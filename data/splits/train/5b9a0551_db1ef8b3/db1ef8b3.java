import java.util.*;
import java.io.*;
import java.math.*;

/**
 *
 * @Har_Har_Mahadev
 */

public class C {

	private static long INF = 2000000000000000000L;

	public static void process() throws IOException {

		int n = sc.nextInt();
		PriorityQueue<Integer> odd = new PriorityQueue<Integer>();
		PriorityQueue<Integer> even = new PriorityQueue<Integer>();
		int arr[] = sc.readArray(n);
		long min = INF;
		long sumOdd = 0;
		long sumEven = 0;
		for (int i = 0; i < n; i++) {
			int x = arr[i];
			if (i % 2 == 0) {
				even.add(x);
				sumEven += x;
			} else {
				odd.add(x);
				sumOdd += x;
			}
			if (i > 0) {
				long ans = 0;
				ans += 1l * even.peek() * (n - (even.size()));
				ans += 1l * odd.peek() * (n - (odd.size()));
				ans += sumOdd + sumEven;
				min = Math.min(ans, min);
			}
		}
		System.out.println(min);
	}

	//=============================================================================
	//--------------------------The End---------------------------------
	//=============================================================================

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
		t = sc.nextInt();
		while (t-- > 0) {
			process();
		}
		out.flush();
		out.close();
	}

	

	/////////////////////////////////////////////////////////////////////////////////////////////////////////


	

	

	

	

	

	

	

	

	

	

	/////////////////////////////////////////////////////////////////////////////////////////////////////////

	

	

	
}
