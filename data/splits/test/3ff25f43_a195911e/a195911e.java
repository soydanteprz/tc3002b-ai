import java.util.*;

import java.io.*;
import java.math.*;


public class Main

{ 
	 static FastReader sc=new FastReader(); 
	 static int dp[];
	 static boolean v[];
//	 static int mod=998244353;;
	 static int mod=1000000007;
	 static int max;
	 static int bit[];
	 //static long fact[];
//	 static long A[];
	 static HashMap<Integer,Integer> map;
	 //static StringBuffer sb=new StringBuffer("");
	 //static HashMap<Integer,Integer> map;
	   static PrintWriter out=new PrintWriter(System.out);
	  public static void main(String[] args)
	  {
		 //  StringBuffer sb=new StringBuffer("");
		  int ttt=1;
		       ttt =i();		       
	        outer :while (ttt-- > 0) 
	        {
	        	int n=i();
	        	long A[]=inputL(n);
	        	long B[]=inputL(n);
	        	long C[]=new long[n];
	        	for(int i=0;i<n;i++) {
	        		C[i]=A[i]-B[i]+1;
	        	}
	        	long min=C[n-1];
	        	long ans=0;
	        	long last=A[n-1];
	        	for(int i=n-1;i>=0;i--) {
	        		if(C[i]>min) {
	        			continue;
	        		}
	        		if(A[i]<min) {
	        			long y=last-min+1;
	        			ans+=y*(y+1)/2;
	        			last=A[i];
	        			min=C[i];
	        			continue;
	        		}
	        		
	        		min=C[i];
	        		
	        	}
	        	long y=last-min+1;
    			ans+=y*(y+1)/2;
	        	System.out.println(ans);
	        }
	        //System.out.println(sb.toString());
		     out.close();
	     
	     
	    //CHECK FOR N=1                    //CHECK FOR M=0
        //CHECK FOR N=1                    //CHECK FOR M=0
       	//CHECK FOR N=1
       	
		        
		     
    }
	 
	 
	 
	
	
	  
	 
	 



//static int find(int A[],int a) {
//	  if(A[a]==a)
//		  return a;
//	  return find(A, A[a]);
//}
//FENWICK TREE



//END







static long[] inputL(int n) {
	long A[]=new long[n];
	   for(int i=0;i<n;i++) {
		   A[i]=sc.nextLong();
	   }
	   return A;
   }
















































     static int i() {
    	 return sc.nextInt();
     }
     
       
     
     
  
  
  
  
  
      
      
      
      
     
        
     
}
