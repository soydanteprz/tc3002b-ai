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
	static long mod=1000000007;
	static int D1[],D[],par[];
	static boolean set[];
	static long INF=Long.MAX_VALUE;
	public static void main(String args[])throws IOException
	{  	

		int T=i();
		while(T-->0)
		{
			int N=i();
			long A[]=inputLong(N);
			boolean f=true;
			TreeSet<Long> set=new TreeSet<>();
			for(int i=0; i<N; i++)
			{
				if(i>=2)
				{
					if(A[i]<A[i-1])
					{
						if(set.ceiling(A[i]+1)!=A[i-1])f=false;
					}
					if(A[i]>A[i-1])
					{
						if(set.floor(A[i]-1)!=A[i-1])f=false;
					}
				}
				set.add(A[i]);
			}
			if(f)System.out.println("YES");
			else System.out.println("NO");
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

	

	

	static long[] inputLong(int N)     {
		long A[]=new long[N];
		for(int i=0; i<A.length; i++)A[i]=in.nextLong();
		return A;
	}

	//IO FUNCTIONS END

}
//class pair implements Comparable<pair>{
//	int index; long a;
//	pair(long a,int index)
//	{
//		this.a=a;
//		this.index=index;
//	}
//	public int compareTo(pair X)
//	{
//		if(this.a>X.a)return 1;
//		if(this.a==X.a)return this.index-X.index;
//		return -1;
//	}
//}

//Code For FastReader
//Code For FastReader
//Code For FastReader
//Code For FastReader

