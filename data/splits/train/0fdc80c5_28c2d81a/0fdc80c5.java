import java.io.*;
import java.util.*;

public class E {

	public static void main(String[] args) {
		
		FastScanner scan = new FastScanner();
		int t = scan.nextInt();
		for(int tt = 0;tt<t;tt++) {
			
			int n = scan.nextInt(), k = scan.nextInt();
			int a[] = scan.readArray(k);
			int temp[] = scan.readArray(k);
			int l[] = new int[n];
			int r[] = new int[n];
			Arrays.fill(l, Integer.MAX_VALUE/2);
			Arrays.fill(r, Integer.MAX_VALUE/2);
			for(int i = 0;i<k;i++) l[a[i]-1] = temp[i];
			for(int i = 0;i<k;i++) r[a[i]-1] = temp[i];
			
			for(int i = 1;i<n;i++) l[i] = Math.min(l[i], l[i-1]+1);
			for(int i = n-2;i>=0;i--) r[i] = Math.min(r[i], r[i+1]+1);
			StringBuilder s = new StringBuilder();
			for(int i = 0;i<n;i++) s.append(Math.min(l[i], r[i])+" ");
			System.out.println(s);
		}
	}



	

}
