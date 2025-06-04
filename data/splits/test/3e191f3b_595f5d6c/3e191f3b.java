import java.util.*;
import java.io.*;
import java.util.Arrays;

public class codeforces {
	static int[]av;
	static int[]oc;
	static int[][]memo;
	static int n;
	public static void main(String[] args) throws Exception {
//		int t=sc.nextInt();
//		while(t-->0) {
			n=sc.nextInt();
			int[]a=new int[n];
			int avs=0;
			for (int i = 0; i < a.length; i++) {
				a[i]=sc.nextInt();
				if(a[i]==0)avs++;
			}
			av=new int[avs];
			oc=new int[n-avs];
			for (int i = 0,c=0,j=0; i < a.length; i++) {
				if(a[i]==1) {
					oc[c++]=i;
				}else {
					av[j++]=i;
				}
			}
			memo=new int[oc.length][av.length];
			for(int[]e:memo)Arrays.fill(e, -1);
			pw.println(dp(0,0));
			
//		}
		pw.close();
	}
	
	public static int dp(int i,int j) {
		if(j==av.length) {
			
			return i!=oc.length?(int)1e9:0;
		}
		if(i==oc.length) {
			return 0;
		}
		if(memo[i][j]!=-1) {
			return memo[i][j];
		}
		int take=dp(i+1,j+1)+Math.abs(av[j]-oc[i]);
		int leave=dp(i,j+1);
		return memo[i][j]=Math.min(take, leave);
	}
	
	
	
	
	
	
	
	

	
	
	
	
	
	
	
     
	
	
	
	
	static Scanner sc=new Scanner(System.in);
	static PrintWriter pw=new PrintWriter(System.out);
}
