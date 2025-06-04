import java.io.*;
import java.math.*;
import java.util.*;



public class test {
	 
	   
	  
	   
	   
	    
	 
	   
	
	   public static void main(String args[]) throws Exception { 
		
		   
		   Scanner sc = new Scanner();
		   StringBuilder res = new StringBuilder();
		 
		   int tc = sc.nextInt();
		  
		   while(tc-->0) { 
			
			   int n=sc.nextInt();
				int k=sc.nextInt();
				
				int[] ac=new int[k];
				
				long[] ans=new long[n];
				Arrays.fill(ans, Integer.MAX_VALUE/2);
				
				for(int i=0;i<k;i++) {
					ac[i]=sc.nextInt()-1;
				}
				for(int i=0;i<k;i++) {
					long x = sc.nextLong();
					
					ans[ac[i]] = x;
				}
				
				
				
				for(int i=1;i<n;i++) {
					ans[i]=Math.min(ans[i], ans[i-1]+1);
				}
				
				for(int i=n-2;i>=0;i--) {
					ans[i]=Math.min(ans[i], ans[i+1]+1);
				}
				
				
               for(int i=0;i<n;i++) {
            	   res.append(ans[i]+" ");
               }
               
               res.append("\n");
		   }
			System.out.println(res);
		   
	   }
}
