import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;
import java.util.StringTokenizer;
 
public class Solution {
	 
	 
	public static void main(String[] args) {
		FastReader sc = new FastReader();
       
        int t=sc.nextInt();
       while(t-->0) {
    	   int n=sc.nextInt();
    	   int r[]=new int[n];
    	   int l[]=new int[n];
    	   for(int i=0;i<n;i++) {
    		   r[i]=sc.nextInt();
    	   }
    	   for(int i=0;i<n;i++) l[i]=r[i]-sc.nextInt()+1;
    	   long ans=0,min=l[n-1],max=r[n-1];
    	   
    	   for(int i=n-2;i>=0;i--) {
    		   if(r[i]>=min) min=Math.min(min,l[i]);
    		   else {
    			   ans+=(max-min+1)*(max-min+2)/2;
    			   max=r[i];
    			   min=l[i];
    		   }
    		   
    	   }
    	   ans+=(max-min+1)*(max-min+2)/2;
    	   System.out.println(ans);
    	   
    	   
       }
}	
}
