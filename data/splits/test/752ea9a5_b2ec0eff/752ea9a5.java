import java.io.*;
import java.util.*;

public class PheonixAndTowers
{
	public static void main(String[] args)throws Exception{ new PheonixAndTowers().run();} 
	long mod=1000000000+7;
//	int[][] ar;
	void solve() throws Exception
	{
		
		
		for(int tt=ni();tt>0;tt--){

			//int n = ni();
			
			
			int n = ni();
			int m =ni();
			int x =ni();
			
			int[] a = new int[n];
			for(int i=0;i<n;i++){
				a[i] =ni();
			}
			PriorityQueue<Pair> pq = new PriorityQueue<>();
			
			for(int i=1;i<=m;i++) pq.add(new Pair(i));
			out.println("YES");
			for(int i:a){
				
				Pair r = pq.remove();
				r.sum+=i;
				pq.add(r);
				out.print(r.id+" ");
				
			}
			out.println();
			
			
			
			
			
			
			
		}
		
	}
	
	class Pair implements Comparable<Pair>{
		int id;
		long sum=0;
		public Pair(int i){
			this.id=i;
		}
		
		
		
	}
	
	
	 

	
	/*FAST INPUT OUTPUT & METHODS BELOW*/
	
	private byte[] buf=new byte[1024];
	private int index;
	private InputStream in;
	private int total;
	private SpaceCharFilter filter;
	PrintWriter out;
	
	
	
	
	
	
	
	
	
	 
	
	/* for (1/a)%mod = ( a^(mod-2) )%mod  ----> use expo to calc -->(a^(mod-2)) */
	
	void run()throws Exception
	{
		in=System.in; out = new PrintWriter(System.out);
		solve();
		out.flush();
	}
	private int scan()throws IOException
	{
		if(total<0)
			throw new InputMismatchException();
		if(index>=total)
		{
			index=0;
			total=in.read(buf);
			if(total<=0)
				return -1;
		}
		return buf[index++];
	}
	private int ni() throws IOException 
	{
		int c = scan();
		while (isSpaceChar(c))
			c = scan();
		int sgn = 1;
		if (c == '-') {
			sgn = -1;
			c = scan();
		}
		int res = 0;
		do {
			if (c < '0' || c > '9')
				throw new InputMismatchException();
			res *= 10;
			res += c - '0';
			c = scan();
		} while (!isSpaceChar(c));
		return res * sgn;
	}
	
	
	
	
	
	private boolean isWhiteSpace(int n)
	{
		if(n==' '||n=='\n'||n=='\r'||n=='\t'||n==-1)
			return true;
		return false;
	}
	private boolean isSpaceChar(int c) {
		if (filter != null)
			return filter.isSpaceChar(c);
		return isWhiteSpace(c);
	}
	private interface SpaceCharFilter {
		public boolean isSpaceChar(int ch);
	}
}
