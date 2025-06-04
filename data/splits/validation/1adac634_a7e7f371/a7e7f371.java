import java.io.*;
import java.util.*;


public class cp {
	static int mod=(int)1e9+7;
	
//	static  Reader sc=new Reader();
	static FastReader sc=new FastReader(System.in);
	public static void main(String[] args) throws IOException { 
		long tc=sc.nextLong();
//		Scanner sc=new Scanner(System.in);
//			int tc=1;
			
//			primeSet=new HashSet<>();
//			sieveOfEratosthenes((int)1e6+5);
			while(tc-->0)
			{
				int n=sc.nextInt();
				int k[]=new int[n];
				int h[]=new int[n];
				for(int i=0;i<n;i++)
					k[i]=sc.nextInt();
				for(int i=0;i<n;i++)
					h[i]=sc.nextInt();
				ArrayList<Pair> interval=new ArrayList<Pair>();
				ArrayList<Pair> act=new ArrayList<Pair>();
				for(int i=0;i<n;i++)
					interval.add(new Pair(k[i]-h[i]+1,k[i]));
				Collections.sort(interval);
				// out.println(interval);
				act.add(interval.get(0));
				for(int i=1;i<n;i++)
				{
					Pair p=act.get(act.size()-1);
					if(p.y<interval.get(i).x)
						act.add(interval.get(i));
					else
						p.y=Math.max(p.y, interval.get(i).y);
				}
				// out.println(act);
				long mana=0;
				for(Pair p: act)
				{
					long x=p.y-p.x+1;
					mana+=(x*(x+1))/2;
				}
				out.println(mana);
				
				
				
				
				
				
				
				
				
//				int n=sc.nextInt();
//				long days[]=new long[n];
//				long power[]=new long[n];
//				for (int i = 0; i < power.length; i++) {
//					days[i]=sc.nextLong();
//				}
//				for (int i = 0; i < power.length; i++) {
//					power[i]=sc.nextLong();
//					
//				}
//				
//				long ans=0;
//				for(int i=0;i<n;i++)
//				{
//					if(i==0)
//					{
//						ans+=power[i]*(power[i]+1L)/2L;
//						continue;
//					}
//					
//					long temp=power[i]*(power[i]+1)/2L;
//					long temp2=(power[i-1]+days[i]-days[i-1])*(power[i-1]+days[i]-days[i-1]+1L)/2L;
//					temp2-=power[i-1]*(power[i-1]+1L)/2L;
//					ans+=Math.min(temp, temp2);
////					if(days[i]-days[i-1]<=power[i])
////					{
////						ans+=power[i]*(power[i]+1)/2;
////					}
////					else {
////						ans+=power[i]*(power[i]+1)/2;
////						ans-=power[i-1]*(power[i-1]+1)/2;
////					}
//						
//					
//				}
//				
//				out.println(ans);
			}
			
				
			
			
			out.flush();
			out.close();
			System.gc();
	
	}
	
	
	


	/*
	   ...SOLUTION ENDS HERE...........SOLUTION ENDS HERE...
 */
	
	
	
	
	
	
	
	
	
	
	
    
     
	
	
	
          
	
	
	
	
	 
	// Array to store inverse of 1 to N
	 
	// Array to precompute inverse of 1! to N!
	 
	// Array to store factorial of first N numbers
	 
	// Function to precompute inverse of numbers
	
	 
	// Function to precompute inverse of factorials
	
	 
	// Function to calculate factorial of 1 to N
	
	 
	// Function to return nCr % p in O(1) time
	
	
	
	
	
	
	
	
	
	
	
	
	
	
		
	


	

	//function to find prime factors of n
	
	

	//fenwick tree implementaion
	

	
	

	
	
	
	//Function to find number of set bits
	
 
	
	
	
	
	
	
	
	
	
	
	
	

	
	
	
	
	
	
	
	  
	
	
	 
	
	
	  
	
	
	
	
	
	
	
	
	
	    // function to find last index <= y
    
    
     
	
	 
	 
	
	 
	 

	 

	 
	 
	
	
	
	
//	static class GraphMap{
//		Map<String,ArrayList<String>> graph;
//		GraphMap() {
//			// TODO Auto-generated constructor stub
//			graph=new HashMap<String,ArrayList<String>>();
//			
//		}
//		void addEdge(String a,String b)
//		{
//			if(graph.containsKey(a))
//			this.graph.get(a).add(b);
//			else {
//				this.graph.put(a, new ArrayList<>());
//				this.graph.get(a).add(b);
//			}
//		}
//	}
//	static void dfsMap(GraphMap g,HashSet<String> vis,String src,int ok)
//	{
//		vis.add(src);
//		
//		if(g.graph.get(src)!=null)
//		{
//			for(String each:g.graph.get(src))
//			{
//				if(!vis.contains(each))
//				{
//					dfsMap(g, vis, each, ok+1);
//				}
//			}
//		}
//		
//		cnt=Math.max(cnt, ok);
//	}
//	static void DFS(Graph g, boolean[] visited, int u)
//		{
//	        visited[u]=true;
//	        
//	        for(int i=0;i<g.list[u].size();i++)
//	        {
//	            int v=g.list[u].get(i);
//	            
//	            if(!visited[v])
//	            {
//	            	cnt1=cnt1*2;
//	            	DFS(g, visited, v);
//	            
//	            }
//	            
//	        }
//	       
//
//	  } 
	
	
	  
	 static class Pair implements Comparable<Pair>
	    {
	       int x;
	       int y;
	       Pair(int x,int y)
	        {
	           this.x=x;
	           this.y=y;
	          
	        }
		
	       
	       
	       
	    }
	   
	
	


	
	
//	 static long modInverse(long a, long m)
//		    {
//		        long g = gcd(a, m);
//		       
//		          return   power(a, m - 2, m);
//		        
//		    }
	
	
	 
	 

	   
	   
	    
	  static  PrintWriter out=new PrintWriter(System.out);

}
