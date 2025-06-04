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

			Integer[] k = intInput(n, in), h = intInput(n, in);
			long ans = 0;
			int a = k[n - 1];
			int current = k[n - 1];
			for (int i = n - 1; i >= 0; i--) {
				if (current > k[i]) {
					ans += sum(a - current + 1);
					a = k[i];
					current = k[i]-h[i]+1;
				}else {
					current = Math.min(current, k[i] - h[i]+1);
				}
			}
			ans += sum(a - current + 1);
			out.println(ans);
		}

		out.flush();
	}

	static long sum(long a) {
		return a * (a + 1) / 2;
	}

	

	

	

	static Integer[] intInput(int n, InputReader in) {
		Integer[] a = new Integer[n];
		for (int i = 0; i < a.length; i++)
			a[i] = in.nextInt();
		return a;
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








