import java.io.*;
import java.util.*;

public class Solution {

	

	

	static int mod = (int) 1e9 + 7;

	//static Scanner sc = new Scanner(System.in);
	static StringBuilder out = new StringBuilder();

	
	static int pri[] = new int[(int) 1e5 + 5];
	
	
	public static void main(String[] args) throws IOException {

		int t = sc.nextInt();

		//sieve();
		int tc = 1;
		while (tc <= t) {

			// out.append("Case #" + tc + ": ");
			Solution run = new Solution();

			run.run();

			tc++;
		}

		System.out.println(out);

	}

	
	ArrayList<Integer> gr[];
	long a[][];
	

	public void run() throws IOException {

		int n = sc.nextInt();
		
		gr=new ArrayList[n+1];
		a=new long[n+1][2];
		for(int i=0;i<=n;i++)gr[i]=new ArrayList<>();
		
	
		for(int i=1;i<=n;i++) {
			
			a[i][0]=sc.nextInt();
			a[i][1]=sc.nextInt();
		}
		
		for(int i=0;i<n-1;i++) {
			
			int u=sc.nextInt();
			int v=sc.nextInt();
			
			gr[u].add(v);
			gr[v].add(u);
		}
		
		dp=new long[n+1][2];
		
		for(long a1[]:dp)Arrays.fill(a1, -1L);
		
		//dfs(1,-1);
		
		long res=Math.max(dfs(1,-1,0), dfs(1,-1,1));
		
		out.append(res+"\n");
		
		
	}
	
	long dp[][];
	
	
	long dfs(int u, int pa,int ok) {
		
		
		
			
			if(dp[u][ok]!=-1)return dp[u][ok];
			
			
		
		
		
		long ans=0;
		for(int ch: gr[u]) {
			
			
			if(ch==pa)continue;
			long res=0;
			
			
				
				if(ok==0) {
					
					res=Math.max(dfs(ch,u,0)+Math.abs(a[u][0]-a[ch][0]), dfs(ch,u,1)+Math.abs(a[u][0]-a[ch][1]));
					
				}
				else {
					
					res=Math.max(dfs(ch,u,0)+Math.abs(a[u][1]-a[ch][0]), dfs(ch,u,1)+Math.abs(a[u][1]-a[ch][1]));
					
					
				}
			
			
			
			ans+=res;
			
			
			
			
		}
		
		return dp[u][ok]=ans;
	}
	
	
	void dfs(int u, int pa) {
		
		if(gr[u].size()==1) {
			dp[u][0]=dp[u][1]=0L;
		}
		
		
		for(int ch:gr[u]) {
			
			
			if(ch==pa)continue;
			
			
			dfs(ch,u);
		}
		
		
		
		for(int ch: gr[u]) {
			
			if(ch==pa)continue;
			
			long ans1=0;
			long ans2=0;
			
			
			ans1=Math.max(dp[ch][0]+Math.abs(a[u][0]-a[ch][0]), dp[ch][1]+Math.abs(a[u][0]-a[u][1]));
			
			dp[u][0]+=ans1;
			
			ans1=Math.max(dp[ch][0]+Math.abs(a[u][1]-a[ch][0]), dp[ch][1]+Math.abs(a[u][1]-a[u][1]));
			
			dp[u][1]+=ans2;
			
			
			
		}
		
		
	}

	

	

	static Reader sc = new Reader();

	

}
