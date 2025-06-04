import java.util.*;
import java.util.Map.Entry;
import java.lang.*;
import java.io.*;
import java.math.BigInteger;


public class CF {
	private static FS sc = new FS();
	
	
	
	private static class extra {
		
		static int[] intArr(int size) {
			int[] a = new int[size];
			for(int i = 0; i < size; i++) a[i] = sc.nextInt();
			return a;
		}
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
	}
	
	static int mod = (int) 1e9 + 9;
//	static int mod = (int) 998244353;
	static int max = (int) 1e6, sq = 316;
	static LinkedList<Integer>[] temp;
//	static int[] par, rank;
	
	public static void main(String[] args) {	
		int t = sc.nextInt();
//		int t = 1;
		StringBuilder ret = new StringBuilder();
		while(t-- > 0) {
			int n = sc.nextInt(); int m = sc.nextInt();
			int[] a = extra.intArr(m);
			int[] b = extra.intArr(m);
			long[] c = new long[n];
			Arrays.fill(c, (int)1e18);
			for(int i = 0; i < m; i++) c[a[i]-1] = b[i];
			long[] l = new long[n];
			long[] r = new long[n];
			Arrays.fill(l, (int)1e18);
			Arrays.fill(r, (int)1e18);
			long min = (long)1e18;
			for(int i = 0; i < n; i++) {
				min = Math.min(min+1, c[i]);
				l[i] = min;
			}
			min = (int)1e18;
			for(int i = n-1; i >= 0; i--) {
				min = Math.min(min+1, c[i]);
				r[i] = min;
			}
			for(int i = 0; i < n; i++) {
				ret.append(Math.min(l[i], r[i]) + " ");
			}
			ret.append("\n");
 		}
		System.out.println(ret);
	}
	
}
