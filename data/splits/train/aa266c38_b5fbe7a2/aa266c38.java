import java.util.*;
import java.io.*;
import java.math.*;

public class Main

{ 
	 static FastReader sc=new FastReader(); 
	 static long dp[][];
	 static int mod=1000000007;

	 
	  public static void main(String[] args)
{
		   PrintWriter out=new PrintWriter(System.out);
		   StringBuffer sb=new StringBuffer("");
		  int ttt=1;
		ttt =i(); 	
		   
	        outer :while (ttt-- > 0) 
			{
	        	int n=i();
	        	int m=i();
	        	int x=i();
	        	Pair A[]=new Pair[n];
	        	for(int i=0;i<n;i++) {
	        		A[i]=new Pair(i(), i+1);
	        	}
	        	Arrays.sort(A);
	        	int B[]=new int[n+1];
	        	int a=m;
	        	for(int i=n-1;i>=0;i--) {
	        		B[A[i].y]=a;
	        		a--;
	        		if(a==0) {
	        			a=m;
	        		}
	        	}
	        	out.println("YES");
	        	for(int i=1;i<=n;i++) {
	        		out.print(B[i]+" ");
	        	}
	        	out.println();
	        	
	        	
			}

	     //System.out.println(sb.toString());
		     out.close();
	     
	     
	    //CHECK FOR N=1                    //CHECK FOR M=0
        //CHECK FOR N=1                    //CHECK FOR M=0
       	//CHECK FOR N=1
       	//CHECK FOR N=1
       	//CHECK FOR N=1
		        
		     
    }
	  
	  
	  
	  
	  
	  
	  
	  
	  
	 
static class Pair implements Comparable<Pair>
     {
    	 int x;
    	 int y;
    	 Pair(int x,int y){
    		 this.x=x;
    		 this.y=y;
    	 }
		
	
		
     }




















static void print(int A[]) {
	for(int i : A) {
		System.out.print(i+" ");
	}
	System.out.println();
}
static void print(long A[]) {
	for(long i : A) {
		System.out.print(i+" ");
	}
	System.out.println();
}



     static int i() {
    	 return sc.nextInt();
     }
     
       
     static void sort(int[] A){
        int n = A.length;
        Random rnd = new Random();
        for(int i=0; i<n; ++i){
            int tmp = A[i];
            int randomPos = i + rnd.nextInt(n-i);
            A[i] = A[randomPos];
            A[randomPos] = tmp;
        }
        Arrays.sort(A);
     }
     static void sort(long[] A){
	        int n = A.length;
	        Random rnd = new Random();
	        for(int i=0; i<n; ++i){
	            long tmp = A[i];
	            int randomPos = i + rnd.nextInt(n-i);
	            A[i] = A[randomPos];
	            A[randomPos] = tmp;
	        }
	        Arrays.sort(A);
	     }
  static String sort(String s) {
 	 Character ch[]=new Character[s.length()];
 	 for(int i=0;i<s.length();i++) {
 		 ch[i]=s.charAt(i);
 	 }
 	 Arrays.sort(ch);
 	 StringBuffer st=new StringBuffer("");
 	 for(int i=0;i<s.length();i++) {
 		 st.append(ch[i]);
 	 }
 	 return st.toString();
  }
  
  
      
      
      
      
     
        
     
}
