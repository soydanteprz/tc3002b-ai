import java.util.*;
import java.io.*;
import java.math.*;


public class Main

{ 
	 static FastReader sc=new FastReader(); 
	 static long dp[][];

	   static PrintWriter out=new PrintWriter(System.out);
	  public static void main(String[] args)
{
		    //CHECK FOR N=1
	       	//CHECK FOR N=1
		   //StringBuffer sb=new StringBuffer("");
		  int ttt=1;
		      ttt =i();
	        outer :while (ttt-- > 0) 
			{
	        	int n=i();
	        	int k=i();
	        	int P[]=input(k);
	        	int Q[]=input(k);
	        	long A[]=new long[n+1];
	        	Arrays.fill(A, Integer.MAX_VALUE);
	        	for(int i=0;i<k;i++) {
	        		A[P[i]]=Q[i];
	        	}
	        	long B[]=new long[n+1];
	        	long C[]=new long[n+1];
	        	Arrays.fill(B, Integer.MAX_VALUE);
	        	Arrays.fill(C, Integer.MAX_VALUE);
	        	for(int i=n;i>0;i--) {
	        		if(i+1<=n) {
	        			B[i]=Math.min(A[i],B[i+1]+1);
	        		}
	        		else {
	        			B[i]=A[i];
	        		}
	        	}
	        	for(int i=1;i<=n;i++) {
	        		if(i-1>0) {
	        			C[i]=Math.min(A[i],C[i-1]+1);
	        		}
	        		else {
	        			C[i]=A[i];
	        		}
	        	}
	        	for(int i=1;i<=n;i++) {
	        		out.print(Math.min(B[i],C[i])+" ");
	        	}
	        	out.println();
	        	
	        	
	        	
	        	
	        	
	        }
	        //System.out.println(sb.toString());
		     out.close();
	     
	     
	    //CHECK FOR N=1                    //CHECK FOR M=0
        //CHECK FOR N=1                    //CHECK FOR M=0
       	//CHECK FOR N=1
       	
		        
		     
    }
	  
	  
	  
	  
	  

//FENWICK TREE



//END
//static void add(int v) {
//	if(!map.containsKey(v)) {
//		map.put(v, 1);
//	}
//	else {
//		map.put(v, map.get(v)+1);
//	}
//}
//static void remove(int v) {
//	if(map.containsKey(v)) {
//		map.put(v, map.get(v)-1);
//		if(map.get(v)==0)
//			map.remove(v);
//	}
//}




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
     
       
     
     
  
  
  
      
      
      
      
     
        
     
}
