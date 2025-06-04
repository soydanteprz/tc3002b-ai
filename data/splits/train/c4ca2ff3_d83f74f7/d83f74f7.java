import java.util.*;

import java.io.*;
import java.math.*;


public class A		
{ 
			 static FastReader sc=new FastReader(); 
			 static long ans;
			 static HashMap<Integer,Integer> map;
			
				
		     public static void main(String[] args)
		    {
		    	//CHECK FOR N=1
		        	//CHECK FOR N=1
			  PrintWriter out=new PrintWriter(System.out);
			  StringBuffer sb=new StringBuffer("");
			 
		  int ttt=1;
		     ttt =i();
		   
		    outer :while (ttt-- > 0) 
			{	
		    	int n=i();
		    	String A[]=inputS(n);
		    	Pair P[]=new Pair[n];
		    	int B[][]=new int[n][5];
		    	for(int i=0;i<n;i++) {
		    		for(int j=0;j<A[i].length();j++) {
		    			B[i][A[i].charAt(j)-'a']++;
		    		}
		    	}
		    	int ans=Integer.MAX_VALUE;
		    	for(int k=0;k<5;k++) {
		    		int s1=0;
		    		int s2=0;
		    	for(int i=0;i<n;i++) {
		    		int a=B[i][k];
		    		int b=A[i].length()-a;
		    		s1+=a;
		    		s2+=b;
		    		P[i]=new Pair(a, b);
		    	}
		    	Arrays.sort(P);
		    	int res=go(P,s1,s2);
		    	ans=Math.min(ans,res);
		    	
		    	}
		    	System.out.println(n-ans);
		    	
		    			
		    	
		    	
			}
		   out.close();
		  // System.out.println(sb.toString());
		   
		   
		   //CHECK FOR N=1                    //CHECK FOR M=0
		    //CHECK FOR N=1                    //CHECK FOR M=0
		   
    }
		 static int go(Pair P[],int a,int b) {
			 int i=0;
			 int ans=0;
			 while(a<=b&& i<P.length) {
				 a-=P[i].x;
				 b-=P[i].y;
				 ans++;
				 i++;
			 }
			 return ans;
		 }


static class Pair implements Comparable<Pair>
     {
    	 int x;
    	 int y;
    	 Pair(int x,int y){
    		 this.x=x;
    		 this.y=y;
    		 
    	 }
		
		
		/* FOR TREE MAP PAIR USE */
		
//		public int compareTo(Pair o) {
// 			if (x > o.x) {
// 				return 1;
// 			}
// 			if (x < o.x) {
// 				return -1;
// 			}
// 			if (y > o.y) {
// 				return 1;
// 			}
// 			if (y < o.y) {
// 				return -1;
// 			}
// 			return 0;
// 		}
//		public int hashCode() 
//		{ 
//			final int temp = 14; 
//			int ans = 1; 
//			ans =x*31+y*13; 
//			return ans; 
//		} 
//  
//		// Equal objects must produce the same 
//		// hash code as long as they are equal 
//		@Override
//		public boolean equals(Object o) 
//		{ 
//			if (this == o) { 
//				return true; 
//			} 
//			if (o == null) { 
//				return false; 
//			} 
//			if (this.getClass() != o.getClass()) { 
//				return false; 
//			} 
//			Pair other = (Pair)o; 
//			if (this.x != other.x || this.y!=other.y) { 
//				return false; 
//			} 
//			return true; 
//		} 	
		
     }
//FRENWICK TREE



//END






static String[] inputS(int n) {
	String A[]=new String[n];
	   for(int i=0;i<n;i++) {
		   A[i]=sc.next();
	   }
	   return A;
   }
















     static int i() {
    	 return sc.nextInt();
     }
     
  
      
      
      
      
     
        
     
}
