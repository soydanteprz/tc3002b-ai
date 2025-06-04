import java.io.*;
import java.util.*;
import java.math.*;
import java.math.BigInteger; 
//import javafx.util.*; 
public final class B
{    
	static StringBuilder ans=new StringBuilder();
	static FastReader in=new FastReader();
	static ArrayList<ArrayList<Integer>> g;
	static long mod=(long)(1e9+7);
	static int D1[],D2[],par[];
	static boolean set[];
	static int value[];
	static long INF=Long.MAX_VALUE;
	static int dp[][];
	static int N,M;
	static int A[][],B[][];
	static int s=1;
	public static void main(String args[])throws IOException
	{  					
		int N=i();
		int A[]=input(N);
		ArrayList<Integer> one=new ArrayList<Integer>();
		ArrayList<Integer> zero=new ArrayList<Integer>();
		for(int i=1; i<=N; i++)
		{
			if(A[i-1]==1)one.add(i);
			else zero.add(i);
		}
		int sum[][]=new int[N+5][N+5];
		for(int i=1; i<=one.size(); i++)
		{
			for(int j=1; j<=zero.size(); j++)
			{
				sum[i][j]=Math.abs(one.get(i-1)-zero.get(j-1));
			}
			//print(sum[i]);
		}
		dp=new int[N+5][N+5];
		//for(int d[]:dp)Arrays.fill(d, Integer.MAX_VALUE);
		Arrays.fill(dp[0], 0);
		for(int i=1; i<=one.size(); i++)
		{
			for(int j=i; j<=zero.size(); j++)
			{
				if(i==j)
				{
					dp[i][j]=dp[i-1][j-1]+sum[i][j];
				}
				else
				{
					dp[i][j]=Math.min(dp[i][j-1], dp[i-1][j-1]+sum[i][j]);
				}
			}
		}
		System.out.println(dp[one.size()][zero.size()]);
//		f(0,0,one,zero,0);
		//for(int d[]:dp)print(d);
		
	}	
	
	
	


	

	
	
	
	
	

	
	
	
	

	static long min(long a,long b,long c)
	{
		return Math.min(a, Math.min(c, b));
	}

	
	
	
	
	

	
	

	
	
	
	
	
	

	//Debugging Functions Starts

	
	
	
	
	

	//Debugging Functions END
	//----------------------
	//IO FUNCTIONS STARTS
	
	static int i()
	{
		return in.nextInt();
	}

	

	static int[] input(int N){
		int A[]=new int[N];
		for(int i=0; i<N; i++)
		{
			A[i]=in.nextInt();
		}
		return A;
	}

	

	//IO FUNCTIONS END

}



//Code For FastReader
//Code For FastReader
//Code For FastReader
//Code For FastReader

