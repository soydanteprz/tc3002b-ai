import java.util.*;


import java.io.*;
import java.math.*;

public class Main

{ 
	 static FastReader sc=new FastReader(); 
	 static long dp[][];
	 static int max;
	  public static void main(String[] args)
{
		   PrintWriter out=new PrintWriter(System.out);
		   //StringBuffer sb=new StringBuffer("");
		  int ttt=1;
		   //ttt =i();
		 
	        outer :while (ttt-- > 0) 
			{
	        	int n=i();
	        	int A[]=input(n);
	        	dp=new long[n+1][n+1];
	        	for(int i=0;i<=n;i++) {
	        		Arrays.fill(dp[i],-1);
	        	}
		        	
	        	
	        	ArrayList<Integer> l=new ArrayList<Integer>();
	        	ArrayList<Integer> m=new ArrayList<Integer>();
	        	for(int i=0;i<n;i++) {
	        		if(A[i]==0) {
	        			l.add(i+1);
	        		}
	        		else {
	        			m.add(i+1);
	        		}
	        	}
	        	A=new int[m.size()];
	        	int B[]=new int[l.size()];
	        	for(int i=0;i<l.size();i++) {
	        		B[i]=l.get(i);
	        	}
	        	for(int i=0;i<m.size();i++) {
	        		A[i]=m.get(i);
	        	}
	        	n=m.size();
	        	int o=l.size();
		        	System.out.println(go(A,B,0,0,n,o));
	        	
			}

	     //System.out.println(sb.toString());
		     out.close();
	     
	     
	    //CHECK FOR N=1                    //CHECK FOR M=0
        //CHECK FOR N=1                    //CHECK FOR M=0
       	//CHECK FOR N=1
       	//CHECK FOR N=1
       	//CHECK FOR N=1
		        
		     
    }
	  
	 
	  
	  
	
	 
	private static long go(int[] A, int[] B, int i, int j, int n, int m) {
		if(i==n)
			return 0;
		if(j==m)
			return Integer.MAX_VALUE;
		if(dp[i][j]!=-1)
			return dp[i][j];
		long op1=go(A, B, i+1, j+1, n, m)+Math.abs(A[i]-B[j]);
		long op2=go(A, B, i, j+1, n, m);
		return dp[i][j]=Math.min(op1, op2);
	
		
		
		
		
		}








static int[] input(int n) {
	int A[]=new int[n];
	   for(int i=0;i<n;i++) {
		   A[i]=sc.nextInt();
	   }
	   return A;
   }

















     static int i() {
    	 return sc.nextInt();
     }
     
     static long l() {
    	 return sc.nextLong();
     }  
     
     
  
  
  
      
      
      
      
     
        
     
}
