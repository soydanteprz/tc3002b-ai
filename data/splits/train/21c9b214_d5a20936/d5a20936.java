import java.util.*;
import java.io.*; 
  
public class Main {
	 
	static long mod = (long)(1e9+7); 
//	 static long mod = 998244353; 
//	 static Scanner sc = new Scanner(System.in); 
	static FastReader sc = new FastReader(); 
	static PrintWriter out = new PrintWriter(System.out);
	static int ans;
	public static void main (String[] args) {
		int ttt = 1;
		ttt = sc.nextInt();
		z :for(int tc=1;tc<=ttt;tc++){
			int n = sc.nextInt();
			char a[] = sc.next().toCharArray();
			char b[] = sc.next().toCharArray();
			
			if(same(a, b)) {
				out.write("0\n");
				continue;
			}
			
			int e1 = 0, e0 = 0, d1 = 0, d0 = 0;
			
			for(int i=0;i<n;i++) {
				if(a[i]!=b[i]) {
					if(a[i] == '0') d0++;
					else d1++;
				}
				else {
					if(a[i] == '0') e0++;
					else e1++;
				}
			}
			
			int ans = 0;
			
//			if(e1 + e0 <= d1 + d0) {
				ans = solveByFirstD(e1, e0, d1, d0);
//			}
//			else {
				ans = Math.min(ans, solveByFirstE(e1, e0, d1, d0));
//			}
			
			if(ans == Integer.MAX_VALUE) {
				ans = -1;
			}
			
			out.write(ans+"\n");
		}
		out.close();
	}
	private static int solveByFirstD(int e1, int e0, int d1, int d0) {
		boolean is = false;
		int cnt = 0;
		while(true) {
			if(d1 == 0 && d0 == 0) return cnt;
			if((is && e1 == 0) || (!is && d1 == 0)) return Integer.MAX_VALUE;
			
			if(is) {
				int td1 = e0;
				int td0 = e1 - 1;
				int te1 = d0 + 1;
				int te0 = d1;
				
				d1 = td1;
				d0 = td0;
				e1 = te1;
				e0 = te0;
			}
			else {
				int td1 = 1 + e0;
				int td0 = e1;
				int te1 = d0;
				int te0 = d1-1;
				
				d1 = td1;
				d0 = td0;
				e1 = te1;
				e0 = te0;
			}
			
			is = !is;
			cnt++;
		}
	}
	private static int solveByFirstE(int e1, int e0, int d1, int d0) {
		boolean is = true;
		int cnt = 0;
		while(true) {
			if(d1 == 0 && d0 == 0) return cnt;
			if((is && e1 == 0) || (!is && d1 == 0)) return Integer.MAX_VALUE;
			
			if(is) {
				int td1 = e0;
				int td0 = e1 - 1;
				int te1 = d0 + 1;
				int te0 = d1;
				
				d1 = td1;
				d0 = td0;
				e1 = te1;
				e0 = te0;
			}
			else {
				int td1 = 1 + e0;
				int td0 = e1;
				int te1 = d0;
				int te0 = d1-1;
				
				d1 = td1;
				d0 = td0;
				e1 = te1;
				e0 = te0;
			}
			
			is = !is;
			cnt++;
		}
	}
	private static boolean same(char[] a, char[] b) {
		for(int i=0;i<a.length;i++) {
			if(a[i]!=b[i]) return false;
		}
		return true;
	}
	
	 
	 
	 
	 
	 
}
