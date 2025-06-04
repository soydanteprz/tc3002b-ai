import java.io.*;
import java.util.*;


public class Practice
{
//	static final long mod=7420738134811L;
	static int mod=1000000007;
	static final int size=501;
	static FastReader sc=new FastReader(System.in);
//	static Reader sc=new Reader();
	static  PrintWriter out=new PrintWriter(System.out);
	static long[] factorialNumInverse;
	static long[] naturalNumInverse;	
	static int[] sp;
	static long[] fact;
	static ArrayList<Integer> pr;
	public static void main(String[] args) throws IOException
	{
//		System.setIn(new FileInputStream("input.txt"));
//		System.setOut(new PrintStream("output.txt"));
//		factorial(mod);
//		InverseofNumber(mod);
//		InverseofFactorial(mod);
//		make_seive();
		int t=1;
		t=sc.nextInt();
		while(t-->0)
			solve();
		out.close();
		out.flush();
	}
	static void solve() throws IOException
	{
		int n=sc.nextInt();
		int arr[]=new int[n];
		for(int i=0;i<n;i++)
			arr[i]=sc.nextInt();
		String s=sc.next();
		ArrayList<Integer> blue=new ArrayList<Integer>();
		ArrayList<Integer> red=new ArrayList<Integer>();
		for(int i=0;i<n;i++)
		{
			if(s.charAt(i)=='B')
				blue.add(arr[i]);
			else
				red.add(arr[i]);
		}
		Collections.sort(blue);
		Collections.sort(red);
		for(int i=0;i<blue.size();i++)
		{
			if(blue.get(i)<i+1)
			{
				out.println("NO");
				return;
			}
		}
		for(int i=0;i<red.size();i++)
		{
			if(red.get(i)>i+1+blue.size())
			{
				out.println("NO");
				return;
			}
		}
		out.println("YES");
	}
	
	
	
	
	
	
	
	// Function to precompute inverse of factorials
	
	// Function to calculate factorial of 1 to 200001
	
	 
	// Function to return nCr % p in O(1) time
	
	
	
	
	
	
	
	
	
	
	
}
