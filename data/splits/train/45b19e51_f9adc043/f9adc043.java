import java.util.*;
import java.io.*;

public class C {
	public static void main(String[] args) throws IOException {
		FastScanner in = new FastScanner(System.in);
		PrintWriter out = new PrintWriter(new BufferedWriter(new OutputStreamWriter(System.out)));
		int bruh = in.nextInt();

		for (int cases = 0; cases < bruh; cases++) {
			int n = in.nextInt();
			long lo = (int) 1e9, hi = 0;
			long [] a = new long[n];
			
			for (int i = 0; i < n; i++) {
				a[i] = in.nextInt();
				lo = Math.min(lo, a[i]);
				hi = Math.max(hi, a[i]);
			}
			
			long m = 0;
			//System.out.println(Arrays.toString(a));
			
			while (lo <= hi) {
				m = (lo + hi) / 2;
				if (works(a, m)) {
					lo = m + 1;
				}
				else {
					hi = m - 1;
				}
			}
			
			out.println(hi);
		}

		out.close();

	}


	static boolean works(long[] a, long m) {
		long[] d = Arrays.copyOf(a, a.length);
		
		for (int i = a.length - 1; i > 1; i--) {
			if (d[i] < m) {
				//System.out.println(m + ":x" + Arrays.toString(d));
				return false;
			}
			long canGive = (d[i] - m) / 3 * 3;
			canGive = Math.min(canGive, a[i]);
			d[i - 1] += canGive / 3;
			d[i - 2] += canGive / 3 * 2;
			d[i] -= canGive;
		}
		
		//System.out.println(m + ": " + Arrays.toString(d));
		
		if (d[0] < m || d[1] < m) return false;
		return true;
	}
	
	
}
