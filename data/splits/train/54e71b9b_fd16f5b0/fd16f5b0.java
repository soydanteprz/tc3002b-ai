import java.util.*;
import java.io.*;
import java.math.*;

public class Main

{ 
	 static FastReader sc=new FastReader(); 
	 static long dp[][];
	  public static void main(String[] args)
{
		   PrintWriter out=new PrintWriter(System.out);
		   //StringBuffer sb=new StringBuffer("");
		  int ttt=1;
		  ttt =i();
	        outer :while (ttt-- > 0) 
			{
	        	int n=i();
	        	int B[][]=new int[n][2];
	        	for(int i=0;i<n;i++) {
	        		B[i][0]=i();
	        		B[i][1]=i();
	        	}
	        	ArrayList<Integer> A[]=new ArrayList[n+1];
	        	for(int i=0;i<A.length;i++) {
	        		A[i]=new ArrayList<Integer>();
	        	}
	        	int m=n-1;
	        	dp=new long[n+1][2];
	        	for(int i=0;i<=n;i++)
	        		Arrays.fill(dp[i],-1);
	        	for(int i=0;i<m;i++) {
	        		int a=i();
	        		int b=i();
	        		A[a].add(b);
	        		A[b].add(a);
	        	}
	        	System.out.println(Math.max(dfs(A, 1, -1, 0, B), dfs(A, 1, -1, 1, B)));

	        			
	        	
	        }
	        	
		
		  
		  

	     //System.out.println(sb.toString());
		     out.close();
	     
	     
	    //CHECK FOR N=1                    //CHECK FOR M=0
        //CHECK FOR N=1                    //CHECK FOR M=0
       	//CHECK FOR N=1
       	//CHECK FOR N=1
       	//CHECK FOR N=1
		        
		     
    }

	  private static long dfs(ArrayList<Integer> [] A, int i,int par,int pv,int B[][]) {
			long ans=0;
			long res=B[i-1][pv];
			if(dp[i][pv]!=-1)
				return dp[i][pv];
			for(int child : A[i]) {
				if(child!=par) {
					long op1=dfs(A, child, i, 0, B)+Math.abs(B[child-1][0]-res);
					long op2=dfs(A, child, i, 1, B)+Math.abs(B[child-1][1]-res);
					ans+=Math.max(op1, op2);
				}
			}
			return dp[i][pv]=ans;
			
		}
	 






















     static int i() {
    	 return sc.nextInt();
     }
     
       
     
     
  
  
  
      
      
      
      
     
        
     
}
