import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.StringTokenizer;
public class P4 {
	static int dp[][]=new int[5000+1][5000+1];
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		FastReader sc=new FastReader();
		int t=1;
		//t=sc.nextInt();
		StringBuilder s=new StringBuilder();
		while(t--!=0) {
			int n=sc.nextInt();
			int a[]=new int[n];
			
			for(int i=0;i<=n;i++) {
				for(int j=0;j<=n;j++) {
					P4.dp[i][j]=-1;
				}
			}
			ArrayList<Integer> one=new ArrayList<>();
			ArrayList<Integer> zero=new ArrayList<>();
			for(int i=0;i<n;i++) {
				a[i]=sc.nextInt();
				if(a[i]==0) {
					zero.add(i);
				}else {
					one.add(i);
				}
			}
			Collections.sort(zero);
			Collections.sort(one);
			long ans=sol(0,0,zero.size(),one.size(),a,zero,one);
			System.out.println(ans);
		}
		//System.out.println(s);
	}
	private static long sol(int i, int j, int n, int m,int a[], ArrayList<Integer> zero, ArrayList<Integer> one) {
		//System.out.println(i+" "+j);
		// TODO Auto-generated method stub
		if(j==m) {
			return 0;
		}
		int av=n-i;
		int rem=m-j;
		if(av<rem) {
			return Integer.MAX_VALUE-1;
		}
		if(dp[i][j]!=-1) {
			return dp[i][j];
		}
		long ans1=sol(i+1,j,n,m,a, zero, one);
		long ans2=Math.abs(zero.get(i)-one.get(j))+sol(i+1,j+1,n,m,a, zero, one);
		dp[i][j]=(int) Math.min(ans1, ans2);
		return dp[i][j];
	}
	 
}
