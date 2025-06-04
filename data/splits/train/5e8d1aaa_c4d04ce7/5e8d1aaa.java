import java.io.*;
import java.util.*;

/*

*/


public class E {
	static FastReader sc=null;
	static long max=(long)1e15+7;
	
	public static void main(String[] args) {
		sc=new FastReader();
		PrintWriter out=new PrintWriter(System.out);
		int t=sc.nextInt();
		while(t-->0) {
			int n=sc.nextInt(),k=sc.nextInt();
			int a[]=sc.readArray(k),te[]=sc.readArray(k);
			int curr[]=new int[n];
			Arrays.fill(curr, -1);
			for(int i=0;i<k;i++) {
				a[i]--;
				curr[a[i]]=te[i];
			}
			long min=max;
			long ts[]=new long[n];
			Arrays.fill(ts, max);
			for(int i=n-1;i>=0;i--) {
				if(curr[i]!=-1) {
					min=Math.min(min, curr[i]+i);
				}
				ts[i]=Math.min(ts[i], min-i);
			}
			min=max;
			for(int i=0;i<n;i++) {
				if(curr[i]!=-1) {
					min=Math.min(min, curr[i]-i);
				}
				ts[i]=Math.min(ts[i], min+i);
			}
			for(long e:ts)out.print(e+" ");
			out.println();
			
		}
		out.close();
		
	
	}
	

}
