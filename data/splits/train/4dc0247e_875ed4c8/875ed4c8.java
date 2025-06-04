import java.util.*;
import java.lang.*;
import java.io.*;


public class Main
{
    static PrintWriter out = new PrintWriter(new BufferedOutputStream(System.out));
	public static void main (String[] args) throws java.lang.Exception
	{
		FastReader sc = new FastReader();
		
		int t = sc.nextInt();
		while(t-->0){
		    
		    solve(sc);

		    
		}
		
	}
	
	
	public static void solve(FastReader sc){

	    int n = sc.nextInt();
	    
	    ArrayList<ArrayList<Edge>> graph = new ArrayList<ArrayList<Edge>>();
	    for(int i = 0;i<n;++i){
	        graph.add(new ArrayList<>());
	    }
	    for(int i = 0;i<n-1;++i){
	        int u = sc.nextInt();
	        int v = sc.nextInt();
	        
	        u--;
	        v--;
	        
	        graph.get(u).add(new Edge(v, i));
	        graph.get(v).add(new Edge(u, i));
	        
	        
	    }
	    
	    int start = 0;
	    for(int i = 0;i<n;++i){
	        if(graph.get(i).size()>2){
	            out.println(-1);out.flush();return;
	        }else if(graph.get(i).size()==1){
	            start=i;
	        }
	    }
	    
	    int val=2;
	    int [] wgt = new int[n-1];

	    int curr = graph.get(start).get(0).node;
	    
	    wgt[graph.get(start).get(0).idx] = val;
	    
	    val=5;
	    
	    while(true){
	        ArrayList<Edge> list = graph.get(curr);
	        if(list.size()==1){
	            break;
	        }else{
	            for(Edge el : list){
	                if(wgt[el.idx]==0){
	                    wgt[el.idx] = val;
	                    val = 7-val;
	                    curr = el.node;
	                }
	            }
	            
	        }
	        
	    }
	    
	    
	    for(int el : wgt){
	        out.print(el + " ");
	    }
	    out.println();
	    out.flush();
	    
	}
	
	
	
	
	
	
	static class Edge {
      int node;
      int idx;

      Edge(int src, int nbr) {
         this.node = src;
         this.idx = nbr;
      }
   }
	/*
	int [] arr = new int[n];
	for(int i = 0;i<n;++i){
	    arr[i] = sc.nextInt();
	}
	*/
	
	
		
}
