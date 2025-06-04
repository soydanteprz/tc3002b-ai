import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import static java.lang.Math.*;
import static java.lang.System.out;
import java.util.*;
import java.io.PrintStream;
import java.io.PrintWriter;
public class A {
 
	
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
		
		int t = sc.nextInt();
		while(t-- > 0)
		{
			long INF = 1000000000000000007L;
			int n = sc.nextInt();
			
			long[] c = new long[n+1];
			for(int i = 1;i<=n;i++) c[i] = sc.nextLong();
			
			long ans = INF;
			long mo = INF,  so = 0, co = 0;
			long me = INF,  se = 0, ce = 0;
			
			for(int i=1;i<=n;i++)
			{
				if(i%2 == 1)
				{
					mo = min(mo,c[i]);
					so += c[i];
					co++;
				}
				else
				{
					me = min(me,c[i]);
					se += c[i];
					ce++;
				}
				if(i>=2)
				{
					long x = so + (n - co) * mo + se + (n - ce) * me;
					ans=min(ans,x);
				}
			}
			fout.println(ans);
		}
			
		fout.close();
	}
}
