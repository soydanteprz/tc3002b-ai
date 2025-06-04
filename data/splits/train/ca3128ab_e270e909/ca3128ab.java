import java.io.*;
import java.math.*;
import java.util.*;


public class test {
	 
	   
	  
	    
	 
	   
	
	
	   public static void main(String args[]) throws Exception { 
		
		   
		   Scanner sc = new Scanner();
		   StringBuilder res = new StringBuilder();
		   
		   int tc = sc.nextInt();
		   
		   while(tc-->0) { 
			
			int n = sc.nextInt();
			
			long c[] = new long[n];
			
			for(int i=0;i<n;i++) { 
				c[i] = sc.nextLong();
			}
			
			
			long min_odd = Integer.MAX_VALUE;
			long min_even = Integer.MAX_VALUE;
		    long ans = Long.MAX_VALUE;
		    long sum = 0;
		    
			int cnt1 = n;
			
			int cnt2 = n;
			
			
		    for(int i = 0; i < n; i++){
		        sum += c[i];
		        if(i % 2 == 0){
		            cnt1--;
		            min_odd = Math.min(min_odd, c[i]);
		        }
		        else{
		            cnt2--;
		            min_even = Math.min(min_even, c[i]);
		        }
		        if(i > 0){
		            long temp = sum + (min_odd * cnt1) + (min_even * cnt2);
		            ans = Math.min(ans, temp);
		        }
		 
		    }
		    
		    System.out.println(ans);
		   }
		   
		   System.out.println(res);
	   }
}
