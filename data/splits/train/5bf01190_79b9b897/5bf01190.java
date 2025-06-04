import java.util.*;
	import java.io.*; 
	public class Main{
		//Fast IO class
	         
	    static long MOD=(long)1e9+7;
	    //debug
	    
	    static FastReader sc=new FastReader();
	    static PrintWriter out=new PrintWriter(System.out);  
	    //Global variables and functions
	    //Main function(The main code starts from here)
	    public static void main (String[] args) throws java.lang.Exception {
	    	int test=1;
	    	test=sc.nextInt();
	    	while(test-->0) {
	    		int n=sc.nextInt();
	    		long a[]=new long[n];
	    		for(int i=0;i<n;i++) a[i]=sc.nextLong();
	    		long x[]=new long[n],pre[]=new long[n];
	    		pre[0]=a[0];
	    		pre[1]=a[1];
	    		for(int i=2;i<n;i++) pre[i]=a[i]+pre[i-2];
	    		x[0]=n*a[0];
	    		x[1]=n*a[1];
	    		long m[]= {a[0],a[1]};
	    		for(int i=2;i<n;i++) {
	    			m[i%2]=Math.min(m[i%2],a[i]);

	    			x[i]=(n-(i+2)/2)*m[i%2]+pre[i];
	    			
	    		}
	    		long ans=Long.MAX_VALUE/10;
	    		for(int i=1;i<n;i++) ans=Math.min(ans, x[i]+x[i-1]);
	    		out.println(ans);
	    				
	    	}
	        out.flush();
	        out.close();
	    }
	}
