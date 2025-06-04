import java.util.*;
import java.io.*;

public class Main {
	

	public static FastReader obj = new FastReader();
	public static PrintWriter out = new PrintWriter(System.out);

	

	

	//Cover the small test cases like for n=1 .
	public static class pair {
		int a;
		int b;

		pair(int  x, int y) {
			a = x;
			b = y;
		}
	}

	

	public static int i() {
		return obj.nextInt();
	}

	

	

	public static int[] i(int n) {
		int[] arr = new int[n];
		for (int i = 0; i < n; i++)
			arr[i] = i();
		return arr;
	}

	

	

	

	

	

	public static void sortpair(ArrayList<pair> arr) {
		//ascending just change return 1 to return -1 and vice versa to get descending.
		//compare based on value of pair.a
		arr.sort(new Comparator<pair>() {
			
		});
	}

	// Take of the small test cases such as when n=1,2 etc.
	// remember in case of fenwick tree ft is 1 based but our array should be 0 based.
	// in fenwick tree when we update some index it doesn't change the value to val but it
	// adds the val value in it so remember to add val-a[i] instead of just adding val.
	//in case of finding the inverse mod do it (biexpo(a,mod-2)%mod + mod )%mod
    public static ArrayList<ArrayList<pair>> adj;
    public static int[] ans;
	public static void main(String[] args) {

		int len = i();
		while (len-- != 0) {
			int n = i();
			adj=new ArrayList<>(n+1);
			ans=new int[n];
			int ok=0;
			for(int i=0;i<=n;i++)adj.add(new ArrayList<>());
			for(int i=0;i<n-1;i++)
			{
				int a=obj.nextInt();
				int b=obj.nextInt();
				adj.get(a).add(new pair(b,i));
				adj.get(b).add(new pair(a,i));
				if(adj.get(a).size()>2)ok=1;
				if(adj.get(b).size()>2)ok=1;
			}
			if(ok==1)out.println(-1);
			else
			{
				int[] vis=new int[n+1];
				if(adj.get(1).size()==2)
				{
					vis[adj.get(1).get(1).a]=1;
					dfs(1,vis,11);
					vis[adj.get(1).get(1).a]=0;
					vis[1]=0;
					dfs(1,vis,2);
				}
				else
				{
					dfs(1,vis,11);
				}
				for(int i=0;i<n-1;i++)out.print(ans[i]+" ");
				out.println();
			}
		}
		out.flush();
	}
	public static void dfs(int cur,int[] vis,int c)
	{
		vis[cur]=1;
		for(pair nd:adj.get(cur))
		{
			if(vis[nd.a]==1)continue;
			if(c==2)ans[nd.b]=11;
			else ans[nd.b]=2;
			dfs(nd.a,vis,ans[nd.b]);
		}
	}
}
