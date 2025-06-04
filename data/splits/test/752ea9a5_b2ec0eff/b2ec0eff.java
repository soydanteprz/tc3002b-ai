import java.io.*;
import java.util.*;

public class ques3
{
	public static void main(String[] args)throws Exception{ new ques3().run();} 
	long mod=1000000000+7;
	
	void solve() throws Exception
	{
		for(int ii=ni();ii>0;ii--)
		{
			int n=ni();
			int m=ni();
			int x=ni();
			long h[]=new long[n];
			for (int i = 0; i <n; i++) 
				h[i]=nl();
			int dp[]=new int[n];
			
			PriorityQueue<long[]> pq=new PriorityQueue<long[]>(new Comparator<long[]>() {
				
			});
			for (int i = 1; i <=m; i++) 
				pq.add(new long[] {i,0});
			
			for(int i=0;i<n;i++)
			{
				long tem[]=pq.poll();
				tem[1]+=h[i];
				dp[i]=(int)tem[0];
				pq.add(tem);
			}
			long min=Integer.MAX_VALUE;
			long max=Integer.MIN_VALUE;
			while(!pq.isEmpty())
			{
				long tem[]=pq.poll();
				min=Math.min(min,tem[1]);
				max=Math.max(max,tem[1]);
			}
			if(max-min>x)
			{
				out.println("NO");
			}
			else
			{
				out.println("YES");
				for (int i = 0; i < dp.length; i++) {
					out.print(dp[i]+" ");
				}
				out.println();
			}
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
	private long nl() throws IOException 
	{
		long num = 0;
		int b;
		boolean minus = false;
		while ((b = scan()) != -1 && !((b >= '0' && b <= '9') || b == '-'))
			;
		if (b == '-') {
			minus = true;
			b = scan();
		}
		
		while (true) {
			if (b >= '0' && b <= '9') {
				num = num * 10 + (b - '0');
			} else {
				return minus ? -num : num;
			}
			b = scan();
		}
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
