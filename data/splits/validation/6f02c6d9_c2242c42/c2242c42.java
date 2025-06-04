import java.io.*;
import java.util.*;



     

public class cp_2 {
	
//	static  Reader sc=new Reader();
	static FastReader sc=new FastReader(System.in);
	public static void main(String[] args) throws IOException { 
		long tc=sc.nextLong();
//		Scanner sc=new Scanner(System.in);
//			int tc=1;
			
//			primeSet=new HashSet<>();
//			sieveOfEratosthenes((int)1e5);
			while(tc-->0)
			{
				int n=sc.nextInt();
				int arr[]=new int[n];
				for (int i = 0; i < arr.length; i++) {
					arr[i]=sc.nextInt();
				}
				
				String s=sc.next();
				ArrayList<Integer> blue=new ArrayList<>();
				ArrayList<Integer> red=new ArrayList<>();
				for(int i=0;i<n;i++)
				{
					if(s.charAt(i)=='B')
						blue.add(arr[i]);
					else {
						red.add(arr[i]);
					}
				}
				
				Collections.sort(blue);
				Collections.sort(red,Collections.reverseOrder());
				boolean flag=true;
				for(int i=0;i<blue.size();i++)
				{
					if(blue.get(i)<i+1)
						flag=false;
				}
				
				for(int i=0;i<red.size();i++)
				{
					if(red.get(i)>n-i)
						flag=false;
				}
				
				printYesNo(flag);
				
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
	
 
	
	
	
	
	
	
	
	
	
	
	
	

	
	
	
	
	
	
	
	  
	
	
	 
	
	
	  
	static void printYesNo(boolean condition)
	{
		if (condition) {
			out.println("Yes");
		}
		else {
			out.println("No");
		}
	}
	
	
	
	
	
	
	
	
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
	 
	
	
	  
	 
	   
	
	

	
	
//	 static long modInverse(long a, long m)
//		    {
//		        long g = gcd(a, m);
//		       
//		          return   power(a, m - 2, m);
//		        
//		    }
	
	 
	 

	   
	   
	    
	  static  PrintWriter out=new PrintWriter(System.out);

}
