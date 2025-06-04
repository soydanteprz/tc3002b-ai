import java.io.*;
import java.util.*;
 
public class D 
{	
	static ArrayList<Integer> g[];
	static int[] vis,dist;
	
	public static void main(String[] args) throws IOException
	{	
		
		FastReader f=new FastReader();
		StringBuffer sb=new StringBuffer();
	
		int test=f.nextInt();
		while(test-->0)
		{
			int n=f.nextInt();
			g=new ArrayList[n+1];
			vis=new int[n+1];
			dist=new int[n+1];
			
			int a=f.nextInt();
			int b=f.nextInt();
			
			int da=f.nextInt();
			int db=f.nextInt();
			
			for(int i=0;i<=n;i++)
				g[i]=new ArrayList<>();
			
			for(int i=1;i<n;i++)
			{
				int x=f.nextInt();
				int y=f.nextInt();
				
				g[x].add(y);
				g[y].add(x);
			}
			dfs(a,0);
			if(-dist[a]+dist[b] <= da)
			{
				sb.append("Alice\n");
				continue;
			}
			int nr=1,mx=0;
			for(int i=1;i<=n;i++)
			{
				if(dist[i]>mx)
				{
					mx=dist[i];
					nr=i;
				}
			}
			vis=new int[n+1];
			dist=new int[n+1];
			dfs(nr,0);
			int dm=0;
			for(int i : dist)
				dm=Math.max(dm,i);
			dm--;
			if(dm<= 2*da)
			{
				sb.append("Alice\n");
				continue;
			}
			if(db>2*da)
			{
				sb.append("Bob\n");
				continue;
			}
			else
			{
				sb.append("Alice\n");
				continue;
			}
		}
		System.out.println(sb);
	}	
		
	static void dfs(int n,int par) 
	{	
		dist[n]=dist[par]+1;
		vis[n] = 1;
		for (int child : g[n]) {
			if (vis[child] == 0)
				dfs(child,n);
		}

	}
	
	 
}
