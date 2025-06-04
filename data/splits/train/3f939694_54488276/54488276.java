import java.io.*;
import java.util.*;

public class C {
	static int n;
	public static void main (String[] args) throws IOException {
		FastReader io = new FastReader();
		int t = io.nextInt();
		for (int ii=0; ii<t; ii++) {
			n = io.nextInt();

			String[] arr = new String[n];

			for (int i=0; i<n; i++) {
				String str = io.next();
				arr[i] = str;
			}

			char[] chars = new char[]{'a','b','c','d','e'};

			int ans = -1;

			for (int i=0; i<5; i++) {
				ans = Math.max(ans, solve(arr, chars[i]));
			}
			System.out.println(ans);
		}
	}

	static int solve(String[] arr, char c) {
		//System.out.println("Comparing based on " + c);
		Arrays.sort(arr, new Comp(c));

		int good = 0;
		int total = 0;
		int ret = 0;

		for (int i=0; i<n; i++) {
			//System.out.println(good + " " + total);
			for (int j=0; j<arr[i].length(); j++) {
				if (arr[i].charAt(j) == c) good++;
			}
			total += arr[i].length();
			if (2 * good > total) {
				ret++;
			} else {
				return ret;
			}
		}

		return ret;
	}

	static class Comp implements Comparator<String> {
		char c;

		public Comp (char c) {
			this.c = c;
		}

		
	}
}
