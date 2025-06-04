import java.io.*;
import java.util.*;
 
public class D 
{	
	static long mod=998244353;
	static long[] facts, factInvs;
	
	public static void main(String[] args)throws IOException 
	{	
		FastReader f=new FastReader();
		StringBuffer sb = new StringBuffer();
		
		int n=f.nextInt();
		int k=f.nextInt();
		
		Time time[]=new Time[n];
		for(int i=0;i<n;i++)
		{
			int x=f.nextInt();
			int y=f.nextInt();
			
			time[i]=new Time(x,y);
		}
		Arrays.sort(time);
		precomp();
//		for(Time t : time)
//			System.out.println(t.x+" "+t.y);
		
		int curr=0;
		long ans=0;
		Queue<Integer> pq=new PriorityQueue<>();
		for(int i=0;i<n;i++)
		{	
//			System.out.println("time seg = "+time[i].x+" "+time[i].y);
			int x=time[i].x;
			pq.add(time[i].y);
			curr++;
			while(!pq.isEmpty() && pq.peek()<x)
			{	
				pq.poll();
				curr--;
			}
//			System.out.println("curr = "+curr);

			if(curr<k)
				continue;
			else
			{	
				long add=getNcR(curr-1,k-1)%mod;
				ans=(ans%mod + add)%mod;
//				System.out.println("add = "+add);
			}
		}
		System.out.println(ans);
	}
	
	static void precomp() 
	{
		facts=new long[1000000];
		factInvs=new long[1000000];
		factInvs[0]=facts[0]=1;
		
		for (int i=1; i<facts.length; i++)
			facts[i]=mul(facts[i-1], i);
		
		factInvs[facts.length-1]=modInv(facts[facts.length-1]);
		
		for (int i=facts.length-2; i>=0; i--)
			factInvs[i]=mul(factInvs[i+1], i+1);
	}
	
	static long getNcR(int n, int k) {
		return mul(facts[n], mul(factInvs[k], factInvs[n-k]));
	}
	
	static long mul(long a, long b) {
		return a*b%mod;
	}
	
	static long modInv(long x) {
		return exp(x, mod-2);
	}
	
	static long exp(long base, long e) 
	{
		if (e==0) 
			return 1;
		long half=exp(base, e/2);
		if (e%2==0) 
			return mul(half, half);
		else
			return mul(half, mul(half, base));
	}
	
	static class Time implements Comparable<Time>
	{
		int x,y;
		Time(int x,int y)
		{
			this.x=x;
			this.y=y;
		}
		
	}
	
	 
}
