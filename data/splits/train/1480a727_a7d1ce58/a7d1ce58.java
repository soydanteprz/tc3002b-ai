import java.util.*;
import java.io.*;
 
public class AiseHi {
	static Scanner sc = new Scanner(System.in);
	public static void main (String[] args) {
		Scanner sc = new Scanner(System.in);
		int t = sc.nextInt();
		z : while(t-->0) {
			int n = sc.nextInt();
			long dsasds[] = new long[n];
			for(int i=0;i<n;i++) {
				dsasds[i] = sc.nextLong();
			}
			long dsasdsuyuiyu[] = new long[n];
			for(int i=0;i<n;i++) {
				dsasdsuyuiyu[i] = sc.nextLong();
			}
			
			long dsasdsuyuiyubtrtr[] = new long[n];
			
			for(int i=n-1;i>0;i--) {
				long dif = dsasds[i] - dsasds[i-1];
				long nxt = 0;
				if(i+1<n) nxt = dsasdsuyuiyubtrtr[i+1];
				long cur = Math.max(nxt, dsasdsuyuiyu[i]);
				if(dif >= cur) {
					dsasdsuyuiyubtrtr[i] = 0;
				}
				else {
					dsasdsuyuiyubtrtr[i] = cur-dif;
				}
//				System.out.println(i+" "+dsasdsuyuiyubtrtr[i]);
			}
			
			long dsasdscsacsauyuiyubtrtr = 0, kokopmom = 0;
			
			for(int i=0;i<n;i++) {
				if(i == 0) {
					dsasdscsacsauyuiyubtrtr = dsasdscsacsauyuiyubtrtr + sumFrom(0, dsasdsuyuiyu[i]);
					kokopmom = dsasdsuyuiyu[i];
				}
				else {
					if(dsasdsuyuiyubtrtr[i] == 0) {
						dsasdscsacsauyuiyubtrtr = dsasdscsacsauyuiyubtrtr + sumFrom(0, dsasdsuyuiyu[i]);
						kokopmom = dsasdsuyuiyu[i];
					}
					else {
						if(dsasdsuyuiyubtrtr[i]>kokopmom) {
							dsasdscsacsauyuiyubtrtr = dsasdscsacsauyuiyubtrtr + sumFrom(kokopmom, dsasdsuyuiyubtrtr[i]);
							dsasdscsacsauyuiyubtrtr = dsasdscsacsauyuiyubtrtr + sumFrom(dsasdsuyuiyubtrtr[i], dsasdsuyuiyubtrtr[i] + (dsasds[i]-dsasds[i-1]));
							kokopmom = dsasdsuyuiyubtrtr[i] + dsasds[i] - dsasds[i-1];
						}
						else {
							dsasdscsacsauyuiyubtrtr = dsasdscsacsauyuiyubtrtr + sumFrom(kokopmom, kokopmom + (dsasds[i]-dsasds[i-1]));
							kokopmom = kokopmom + dsasds[i] - dsasds[i-1];
						}
					}
				}
			}
			
			System.out.println(dsasdscsacsauyuiyubtrtr);
		}
	}

	private static long sumFrom(long l, long cur) {
		long is1 = (cur*(cur+1))/2;
		long is2 = (l*(l+1))/2;
		
		return is1 - is2;
	}

	
	
	
	
	
	
	
	
	 
	
	
	
	
  
    
//	private static boolean check(long n,long val) {
//		if(n == 0) return true;
//		
//		while(val<=n) {
////			if(check(n-val,))
//		}
//	}
	
	
	
	
//	static int ceil(int a,int b) {
//		return a/b + (a%b==0?0:1);
//	}
//	static int fac[] = new int[2000009];
	
	
//	static long gcd(long a,long b) { if(b==0) return a; return gcd(b,a%b); }
}
 



