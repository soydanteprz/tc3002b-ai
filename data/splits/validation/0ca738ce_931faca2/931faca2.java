import java.util.*;
import java.io.*;
import java.math.*;



public class Main

{ 
	 static FastReader sc=new FastReader(); 
	 static int dp[][][][];
	 static int mod=1000000007;
	 static int mod1=998244353;
	 static int max;
	 static long bit[];
	 static long seg[];
	 static long fact[];
	 static long A[];
	 static long[] fac = new long[300001];
	   static PrintWriter out=new PrintWriter(System.out);
	  public static void main(String[] args)
{
		    //CHECK FOR N=1
	       	//CHECK FOR N=1
		   //StringBuffer sb=new StringBuffer("");
		  int ttt=1;
		     // ttt =i();
		 
	      fac[0] = 1;

	      for (int i = 1; i <= 300000; i++)
	          fac[i] = fac[i - 1] * i % mod1;
	        outer :while (ttt-- > 0) 
			{
	        	int n=i();
	        	int k=i();
	        	Pair P[]=new Pair[2*n];
	        	int c=0;
	        	for(int i=0;i<n;i++) {
	        		P[c]=new Pair(i(),0);
	        		c++;
	        		P[c]=new Pair(i(),1);
	        		c++;
	        	}
	        	Arrays.sort(P);
	        	int cnt=0;
	        	long ans=0;
	        	for(int i=0;i<2*n;i++) {
	        		if(P[i].y==0)
	        			cnt++;
	        		else {
	        			ans+=nCrModPFermat(cnt-1, k-1, mod1);
	        			ans%=mod1;
	        			cnt--;
	        		}
	        	}

	        	System.out.println(ans);
	        	
	        	
	        }
	        //System.out.println(sb.toString());
		     out.close();
	     
	     
	    //CHECK FOR N=1                    //CHECK FOR M=0
        //CHECK FOR N=1                    //CHECK FOR M=0
       	//CHECK FOR N=1
       	
		        
		     
    }
	  
	  static long modInverse(long n, int p)
	    {
	        return power(n, p - 2, p);
	    }
	 
	    // Returns nCr % p using Fermat's
	    // little theorem.
	    static long nCrModPFermat(int n, int r,
	                             int p)
	    {
	 
	          if (n<r)
	              return 0;
	      // Base case
	        if (r == 0)
	            return 1;
	        
	 
	        return (fac[n] * modInverse(fac[r], p)
	                % p * modInverse(fac[n - r], p)
	                % p)
	            % p;
	    }


static class Pair implements Comparable<Pair>
     {
    	 int x;
    	 int y;
    	 Pair(int x,int y){
    		 this.x=x;
    		 this.y=y;
    		 
    	 }
		
//		public int hashCode() 
//		{ 
//			final int temp = 14; 
//			int ans = 1; 
//			ans =x*31+y*13; 
//			return ans; 
//		} 
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
//		
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












































static long power(long x, long y, long p)
{

	if(y==0)
		return 1;
	if(x==0)
		return 0;
    long res = 1;
    x = x % p;

    while (y > 0) {

        if (y % 2 == 1)
            res = (res * x) % p;

        y = y >> 1; 
        x = (x * x) % p;
    }

    return res;
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
