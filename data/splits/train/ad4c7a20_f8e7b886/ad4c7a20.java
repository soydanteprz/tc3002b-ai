import java.io.*;
import java.util.*;


public class Practice
{
//	static final long mod=7420738134811L;
	static int mod=1000000007;
	static final int size=501;
	static FastReader sc=new FastReader(System.in);
//	static Reader sc=new Reader();
//	static Scanner sc=new Scanner(System.in);
	static  PrintWriter out=new PrintWriter(System.out);
	static long[] factorialNumInverse;
	static long[] naturalNumInverse;	
	static int[] sp;
	static long[] fact;
	static ArrayList<Integer> pr;
	public static void main(String[] args) throws IOException, CloneNotSupportedException
	{
//		System.setIn(new FileInputStream("input.txt"));
//		System.setOut(new PrintStream("output.txt"));
//		factorial(mod);
//		InverseofNumber(mod);
//		InverseofFactorial(mod);
//		make_seive();
		int t=1;
		t=sc.nextInt();
		for(int i=1;i<=t;i++)
			solve(i);
		out.close();
		out.flush();
//		System.out.flush();
//		System.exit(0);
	}
	
	static void solve(int CASENO) throws IOException, CloneNotSupportedException
	{
		int n=sc.nextInt();
		int k[]=new int[n];
		int h[]=new int[n];
		for(int i=0;i<n;i++)
			k[i]=sc.nextInt();
		for(int i=0;i<n;i++)
			h[i]=sc.nextInt();
		ArrayList<Pair> interval=new ArrayList<Pair>();
		ArrayList<Pair> act=new ArrayList<Pair>();
		for(int i=0;i<n;i++)
			interval.add(new Pair(k[i]-h[i]+1,k[i]));
		Collections.sort(interval);
//		out.println(interval);
		act.add(interval.get(0).clone());
		for(int i=1;i<n;i++)
		{
			Pair p=act.get(act.size()-1);
			if(p.y<interval.get(i).x)
				act.add(interval.get(i).clone());
			else
				p.y=Math.max(p.y, interval.get(i).y);
		}
//		out.println(act);
		long mana=0;
		for(Pair p: act)
		{
			long x=p.y-p.x+1;
			mana+=(x*(x+1))/2;
		}
		out.println(mana);
	}
	static class Pair implements Cloneable, Comparable<Pair>
	{
		int x,y;
		Pair(int a,int b)
		{
			this.x=a;
			this.y=b;
		}
		
		
		
		@Override
		protected Pair clone() throws CloneNotSupportedException {
			return new Pair(this.x,this.y);
		}
		
		
	}
	
	
	
	
	
	
	// Function to precompute inverse of factorials
	
	// Function to calculate factorial of 1 to 200001
	
	 
	// Function to return nCr % p in O(1) time
	
	
	
	
	
	
	
	
	
	
	
}
