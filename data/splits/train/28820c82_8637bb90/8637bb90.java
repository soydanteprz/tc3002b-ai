import java.util.*;


import java.awt.Point;
import java.io.*;
import java.math.BigInteger;
public class Solutions {

		static int MAX=Integer.MAX_VALUE;
		static int MIN=Integer.MIN_VALUE;
//static ArrayList<ArrayList<Integer>>list=new ArrayList<ArrayList<Integer>>();
	static FastScanner scr=new FastScanner();
	static PrintStream out=new PrintStream(System.out);
	public static void main(String []args) {
		int T=scr.nextInt();
		t:for(int tt=0;tt<T;tt++) {
			int n=scr.nextInt();
			int []a=scr.readArray(n);
			long min[]=new long[2];  
  			long sum[]=new long[2]; 
			sum  [0]=a[0];
			sum  [1]=0;
			min[0]=a[0];
			min[1]=MAX;
			long ans=Long.MAX_VALUE;
			for(int i=1;i<n;i++) {
				min[i%2]=Math.min(min[i%2],a[i]);
				sum[i%2]+=a[i];
				
				int odd=(i+2)/2;
				int even=(i+1)/2;
				ans=Math.min(ans,sum[0]+((n-odd)*min[0])+sum[1]+((n-even)*min[1]));
			}
			out.println(ans);
		}
	}
	
	
	

}


