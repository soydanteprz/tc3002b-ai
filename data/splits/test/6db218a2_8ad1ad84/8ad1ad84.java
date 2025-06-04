import java.io.*;
import java.util.*;
import java.math.*;
import java.math.BigInteger; 
//import javafx.util.*; 
public final class B
{    
	 static PrintWriter out = new PrintWriter(System.out); 
	static StringBuilder ans=new StringBuilder();
	static FastReader in=new FastReader();
	static ArrayList<Integer> g[];
	static long L[],R[],dp[][];
	public static void main(String args[])throws IOException
	{  					
		int T=i();
		outer:while(T-->0)
		{
			int N=i();
			setGraph(N);
			for(int i=1; i<=N; i++)
			{
				L[i]=l();
				R[i]=l();
			}
			for(int i=1; i<N; i++)
			{
				int a=i(),b=i();
				g[a].add(b);
				g[b].add(a);
			}
			f(1,-1);
			out.println(Math.max(dp[0][1], dp[1][1]));
		}
		out.close();
		//print(L);
		//print(R);
		
		//print(dp[0]);
		//print(dp[1]);
		

	}	
	static void f(int n,int p)
	{
		for(int c:g[n])
		{
			if(c!=p)
			{
			f(c,n);
			long a=dp[0][c]+Math.abs(L[c]-L[n]);
			long b=dp[1][c]+Math.abs(R[c]-L[n]);
			dp[0][n]+=Math.max(a, b);
			a=dp[0][c]+Math.abs(L[c]-R[n]);
			b=dp[1][c]+Math.abs(R[c]-R[n]);
			dp[1][n]+=Math.max(a, b);
			}
		}
	}
	

	
	


	
	

	
	
	
	

	

	
	
	
	
	

	
	static void setGraph(int N)
	{
		g=new ArrayList[N+1];
		dp=new long[2][N+1];
		L=new long[N+1];
		R=new long[N+1];
		for(int i=0; i<=N; i++)
		{

			g[i]=new ArrayList<Integer>();
		}
	}

	
	
	
	
	
	

	//Debugging Functions Starts

	
	
	
	
	

	//Debugging Functions END
	//----------------------
	//IO FUNCTIONS STARTS
	
	static int i()
	{
		return in.nextInt();
	}

	static long l()
	{
		return in.nextLong();
	}

	

	

	//IO FUNCTIONS END

}



//Code For FastReader
//Code For FastReader
//Code For FastReader
//Code For FastReader

