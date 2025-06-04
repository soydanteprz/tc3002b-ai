import java.io.*;
import java.util.*;

public class c {
/*
3
2
13 88
3
2 3 1
5
4 3 2 1 4

202
13
19

+ - + - +
1, 2
2, 1
3, 3
*/
	public static void main(String args[]) throws IOException {
		FastScanner in = new FastScanner(System.in);
		PrintWriter out = new PrintWriter(System.out);
		
		int t = in.nextInt();
		for ( ; t > 0; t--) {
			int n = in.nextInt();
			long[] vals = new long[n];
			for (int i = 0; i < n; i++)
				vals[i] = in.nextInt();
			
				
			// System.out.println(Arrays.toString(pre));
			// sSystem.out.println(Arrays.toString(pre1));
			
			long oo = (long)(1e18);
			long[] min = {oo, oo};
			long[] sub = {n, n};
			long sum = 0;
			long max = oo;
			for (int i = 0; i < n; i++) {
				min[i % 2] = Math.min(min[i % 2], vals[i]);
				sub[i % 2]--;
				sum += vals[i];
				if (i > 0) {
					max = Math.min(max, sum + min[0] * sub[0] + min[1] * sub[1]);
				}
			}
			out.println(max);
		}
		
		
		out.close();
	}
	
	
}
