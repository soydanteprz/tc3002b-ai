import java.util.*;
import java.io.*;

public class codeforces {
	public static void main(String[] args) throws Exception {
		int t=sc.nextInt();
		while(t-->0) {
			int n=sc.nextInt();
			char[] a=sc.next().toCharArray();
			char[] b=sc.next().toCharArray();
			int e0=0;
			int e1=0;
			int o0=0;
			int o1=0;
			for(int i=0;i<n;i++) {
				if(a[i]!=b[i]) {
					if(a[i]=='1') {
						e1++;
					}else {
						e0++;
					}
				}else {
					if(a[i]=='1') {
						o1++;
					}else {
						o0++;
					}
				}
			}
			int ans=Integer.MAX_VALUE;
			
			if(e1==e0) {
				ans=Math.min(ans, e1+e0);
			}
			if(o1==o0+1) {
				ans=Math.min(ans, o1+o0);
			}
//			pw.println(e0+" "+e1+" "+o0+" "+o1);
			pw.println(ans==Integer.MAX_VALUE?-1:ans);
		}
		pw.close();
	}

	

	

	

	static long mod = 1000000007;
	static Random rn = new Random();
	static Scanner sc = new Scanner(System.in);
	static PrintWriter pw = new PrintWriter(System.out);
}
