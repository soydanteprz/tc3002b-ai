import java.io.*;
import java.math.*;
import java.util.*;



public class test {
	 
	   
	  
	    
	 
	   
	
	   
	     
	   public static void main(String args[]) throws Exception { 
		
		   Scanner sc = new Scanner();
		   StringBuffer res = new StringBuffer();
		   
		   int tc = 1;
		   
		   while(tc-->0) { 
			
			    int n = sc.nextInt();
			    
			    ArrayList<Integer> one = new ArrayList<>();
			    ArrayList<Integer> zero = new ArrayList<>();
			    
			    for(int i=0;i<n;i++) { 
			    	
			    	int x = sc.nextInt();
			    	
			    	if(x==1) { 
			    		one.add(i);
			    	}
			    	else { 
			    		zero.add(i);
			    	}
			    }
			    
			    int dp[][] = new int[one.size()+1][zero.size()+1];
			    
			     for(int i=1;i<=one.size();i++)
			     {
			         dp[i][i]=dp[i-1][i-1]+Math.abs(zero.get(i-1)-one.get(i-1));
			         for(int j=i+1;j<=zero.size();j++)
			         {
			             dp[i][j]=Math.min(dp[i][j-1],dp[i-1][j-1]+Math.abs(one.get(i-1)-zero.get(j-1)));
			         }
			     }
			     System.out.println(dp[one.size()][zero.size()]);
		   }
			System.out.println(res);
	   }
}
