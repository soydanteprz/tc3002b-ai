import java.io.*;
import java.util.*;

public class Solution{
    
    public static class Edge
    {
        int index; int u, v;
        Edge(int u, int v, int index)
        {
            this.index=index;
            this.u = u;
            this.v = v;
        }
        
    }
	public static void main (String[] args) throws Exception 
	{
		BufferedReader br=new BufferedReader(new InputStreamReader(System.in));
		StringBuilder sb=new StringBuilder();
		int t=Integer.parseInt(br.readLine());
		
		while(t-->0)
		{
		    int n=Integer.parseInt(br.readLine());
		    ArrayList<Edge> graph[] = new ArrayList[n];
		    for(int i=0; i<n; i++) graph[i] = new ArrayList<>();
		    
		    for(int i=0; i<n-1; i++)
		    {
		        StringTokenizer st=new StringTokenizer(br.readLine());
		        int u = Integer.parseInt(st.nextToken());
		        int v = Integer.parseInt(st.nextToken());
		        
		        u--; v--;
		        graph[u].add(new Edge(u,v,i));
		        graph[v].add(new Edge(v,u,i));
		    }
		    
		    int start_i = 0;
		    boolean poss = true;
		    for(int i =0; i<n; i++)
		    {
		        if(graph[i].size()>2) 
		        {
		            poss = false;
		        }
		        else if(graph[i].size() == 1) start_i = i;
		    }
		    
		    if(!poss) 
		    {
		        sb.append("-1\n");
		        continue;
		    }
		    
		    int wts[] = new int[n-1];
		    
		    ArrayList<Edge> edges = new ArrayList<>();
		    
		    Edge first = graph[start_i].get(0);
		    wts[first.index] = 2;
		    
		    edges = graph[first.v];
		    
		    while(edges.size()>1)
		    {
		        
		        Edge firste = edges.get(0);
		        Edge sece = edges.get(1);
		        if(wts[firste.index] != 0 ) //second not visited
		        {
		            wts[sece.index] = 5 - wts[firste.index];
		            edges = graph[sece.v];
		        }
		        else //first not visited
		        {
		            wts[firste.index] = 5 - wts[sece.index];
		            edges = graph[firste.v];
		        }
		    }
		    
		    for(int i=0; i<n-1; i++)
		        sb.append(wts[i]+" ");
	        sb.append("\n");
		    
		    
		}
		System.out.println(sb);
	}
}
