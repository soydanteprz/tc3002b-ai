import java.io.*;

import java.util.*;
/*


 */



public class C {
    static FastReader sc=null;
 
    
    public static void main(String[] args) {
    	sc=new FastReader();
    	int t=sc.nextInt();
    	while(t-->0) {
    		int n=sc.nextInt();
    		int c[]=sc.readArray(n);
    		long dp[]=new long[n];
    		dp[0]=Long.MAX_VALUE;
    		dp[1]=(long)n*c[1]+ (long)n*c[0];
    		long preMin[]=new long[n],pre[]=new long[n];
    		preMin[0]=pre[0]=c[0];
    		preMin[1]=pre[1]=c[1];
    		for(int i=2;i<n;i++) {
    			preMin[i]=Math.min(preMin[i-2],c[i]);
    			pre[i]=pre[i-2]+c[i];
    		}
    		for(int i=2;i<n;i++) {
    			dp[i]=preMin[i]*(n-i/2)+pre[i]-preMin[i]+preMin[i-1]*(n-(i-1)/2)+pre[i-1]-preMin[i-1];
    		}
    		long ans=Long.MAX_VALUE;
    		//print(dp);
    		for(long e:dp)ans=Math.min(ans, e);
    		System.out.println(ans);
    	}
    	
    	
    }
  
  
   
    
    
   
 
   
    
    
    
    
    
 
       
        
 
    
    
    
    
    
    
    
    
    
    
     
}
