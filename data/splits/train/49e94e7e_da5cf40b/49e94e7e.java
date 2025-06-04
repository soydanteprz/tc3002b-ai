import java.io.BufferedReader;
import java.io.IOException;
import java.lang.*;
import java.io.InputStreamReader;
import static java.lang.Math.*;
import static java.lang.System.out;
import java.util.*;
import java.io.File;
import java.io.PrintStream;
import java.io.PrintWriter;
import java.math.BigInteger;
public class Main {
 
	
	/* 10^(7) = 1s.
	 * ceilVal = (a+b-1) / b */
	
	static final int mod = 1000000007;
	static final long temp = 998244353;
	static final long MOD = 1000000007;
	static final long M = (long)1e9+7;
 
	
	
	
	
	
	
	

	
	
	 
 
		
	
	
	
		
	
		//Count the number of coprime's upto N
			
	
	
	
	   
	   
	   
	 
	    // Returns nCr % p using Fermat's little theorem.
	    
	    
	    
	    
	    
	    
 
		//Modular Operations for Addition and Multiplication. 
		   
		   
		  
		   
		   
		   
		   
		   
			
			
			
			
			
			
			   
				
		   
		   
		
			   /*
			    * 
			    * >=         <=
			   0 1 2 3 4 5 6 7  
			   5 5 5 6 6 6 7 7
			   lower_bound for 6 at index 3 (>=)
			   upper_bound for 6 at index 6(To get six reduce by one) (<=)
			   */

		   
		   
			
			
			
			
			
	
	
	public static void main(String[] args) throws Exception 
	{
		Reader sc = new Reader();
		PrintWriter fout = new PrintWriter(System.out); 
		
		int tt = sc.nextInt();
		while(tt-- > 0)
		{
	       int n = sc.nextInt();
	       char[] a = sc.next().toCharArray(), b = sc.next().toCharArray();
	       
	       int c00 = 0, c01 = 0, c10 = 0, c11 = 0;
	       for(int i = 0;i<n;i++)
	       {
	    	   if(a[i] == '0' && b[i] == '0')
	    	   {
	    		   c00++;
	    	   }
	    	   else if(a[i] == '0' && b[i] == '1')
	    	   {
	    		   c01++;
	    	   }
	    	   else if(a[i] == '1' && b[i] == '0')
	    	   {
	    		   c10++;
	    	   }
	    	   else if(a[i] == '1' && b[i] == '1')
	    	   {
	    		   c11++;
	    	   }
	       }
	       
	       int ans = mod;
	       if(c01 == c10) ans = min(ans, c01 + c10);
	       if(c11 == c00 + 1) ans = min(ans, c11 + c00);
	       
	       fout.println((ans == mod) ? -1 : ans);
		}
		
		fout.close();
	}
}
