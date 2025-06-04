import java.io.*;
import java.util.*;


import java.math.*;
import java.math.BigInteger;


public final class B
{    
	static PrintWriter out = new PrintWriter(System.out); 
	static StringBuilder ans=new StringBuilder();
	static FastReader in=new FastReader();
	static ArrayList<Integer> g[];
	static long mod=(long)998244353,INF=Long.MAX_VALUE;
	//	static boolean set[]; 
	static int par[],partial[];
	static int Days[],P[][];
	static int sum=0,size[];
	static int seg[],col[];
	//	static ArrayList<Long> A;
	static char X[][];
	static boolean set[][];
	static int D[],min[],A[];
	static long dp[][];
	//	static HashSet<Integer> visited,imposters;
	//	static HashSet<Integer> set;
	//	static node1 seg[];
	//static pair moves[]= {new pair(-1,0),new pair(1,0), new pair(0,-1), new pair(0,1)};
	public static void main(String args[])throws IOException
	{	
		/*
		 * star,rope,TPST
		 * BS,LST,MS,MQ
		 */
		int T=i();
		outer:while(T-->0)
		{
			int N=i();
			int f[]=new int[N+5];
			int ask=ask(1);
			HashSet<Integer> set=new HashSet<>();
			int cnt=0;
			for(int i=1; i<=N; i++)
			{
				if(cnt+1==N)
				{
					set=new HashSet<>();
					for( i=0; i<=N; i++)set.add(i);
					for( i=1; i<=N; i++)set.remove(f[i]);
					int a=-1;
					for(int v:set)
					{
						a=v;
					}
					for(i=1; i<=N; i++)
					{
						if(f[i]==0)f[i]=a;
					}
					break;
				}
				if(f[i]==0)
				{
					int last=0;
					set=new HashSet<>();
					while(true)
					{
						int a=ask(i);
						f[last]=a;
						if(set.contains(a))
						{
							break;
						}
						last=a;
						set.add(a);
					}
					cnt+=set.size();
				}
				
			}
			for(int i=1; i<=N; i++)
			{
				ans.append(f[i]+" ");
			}
			System.out.println("! "+ans);
			ans=new StringBuilder();
		}
		out.println(ans);
		out.close();
	}
	static int ask(int a)
	{
		System.out.println("? "+a);
		//		out.flush();
		return i();
	}
	
	
	

	
	
	
	
	
	
	
	
	
	
	
	
	static long fact[];
	
	
	

	
	
	
	
	static int ask(int v,int tl,int tr,int l,int r)
	{
		//		System.out.println(v);
		//		if(v>100)return 0;
		if(l>r)return Integer.MAX_VALUE;
		if(tl==l && tr==r)return seg[v];
		int tm=(tl+tr)/2;
		int a=ask(v*2,tl,tm,l,Math.min(tm, r));
		//		System.out.println("for--> "+(v)+" tm--> "+(tm+1)+" tr--> "+tr+" l--> "+Math.max(l, tm+1)+" r--> "+r);
		int b=ask(v*2+1,tm+1,tr,Math.max(l, tm+1),r);
		return Math.min(a, b);
	}
	
	
	static int ask(int a,int b)
	{
		System.out.println("? "+a+" "+b);
		return i();
	}




	
	
	
	
	
	
	


	//	static void build(int v,int tl,int tr,long A[])
	//	{
	//		if(tl==tr)
	//		{
	//			seg[v]=A[tl];
	//		}
	//		else
	//		{
	//			int tm=(tl+tr)/2;
	//			build(v*2,tl,tm,A);
	//			build(v*2+1,tm+1,tr,A);
	//			seg[v]=Math.min(seg[v*2], seg[v*2+1]);
	//		}
	//	}

	





	
	



	
	
	



	

	    



	 

	

	

	

	  

	



	

	

	

	

	

	

	
	
	
	
	
	
	
	
	

	static int i()
	{
		return in.nextInt();
	}

	

	

	

	

}



//Code For FastReader
//Code For FastReader
//Code For FastReader
//Code For FastReader

