import java.util.*;
import java.io.*; 
  
public class Main {
	 
	static long mod = (long)(1e9+7); 
//	 static long mod = 998244353; 
//	 static Scanner sc = new Scanner(System.in); 
	static FastReader sc = new FastReader(); 
	static PrintWriter out = new PrintWriter(System.out);
	public static void main (String[] args) {
		int ttt = 1;
		ttt = sc.nextInt();
		z :for(int tc=1;tc<=ttt;tc++){
			int n = sc.nextInt();
			long d[] = new long[n];
			for(int i=0;i<n;i++) {
				d[i] = sc.nextLong();
			}
			long h[] = new long[n];
			for(int i=0;i<n;i++) {
				h[i] = sc.nextLong();
			}
			
			long req[] = new long[n];
			
			for(int i=n-1;i>0;i--) {
				long dif = d[i] - d[i-1];
				long nxt = 0;
				if(i+1<n) nxt = req[i+1];
				long cur = Math.max(nxt, h[i]);
				if(dif >= cur) {
					req[i] = 0;
				}
				else {
					req[i] = cur-dif;
				}
//				System.out.println(i+" "+req[i]);
			}
			
			long ans = 0, prev = 0;
			
			for(int i=0;i<n;i++) {
				if(i == 0) {
					ans = ans + sumFrom(0, h[i]);
					prev = h[i];
				}
				else {
					if(req[i] == 0) {
						ans = ans + sumFrom(0, h[i]);
						prev = h[i];
					}
					else {
						if(req[i]>prev) {
							ans = ans + sumFrom(prev, req[i]);
							ans = ans + sumFrom(req[i], req[i] + (d[i]-d[i-1]));
							prev = req[i] + d[i] - d[i-1];
						}
						else {
							ans = ans + sumFrom(prev, prev + (d[i]-d[i-1]));
							prev = prev + d[i] - d[i-1];
						}
					}
				}
			}
			
			out.write(ans+"\n");
		}
		out.close();
	}
	private static long sumFrom(long l, long cur) {
		long is1 = (cur*(cur+1))/2;
		long is2 = (l*(l+1))/2;
		
		return is1 - is2;
	}
	
	 
	 
	 
	 
	 
}
