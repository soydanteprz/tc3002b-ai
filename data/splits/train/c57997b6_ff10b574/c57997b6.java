import java.io.*;
import java.util.*;
import java.util.Map.Entry;

public class AMain {
	
	public static void main(String[] args) {
		FastReader in = new FastReader(System.in);
		
		try(PrintWriter out = new PrintWriter(new BufferedOutputStream(System.out));)
		{
			new AMain().solve(in, out);
		}
	}
	
	public void solve(FastReader in, PrintWriter out)
	{
		int t = in.nextInt();
		while(t-->0)
		{
			int n = in.nextInt();
			int[] a= in.nextInts(n);
			char[] clr = in.next().toCharArray();
			
			ArrayList<ArrayList<Integer> > rgh = new ArrayList<>();
			for(int i=0;i<n;i++)
				rgh.add(new ArrayList<>());
			
			boolean ok = true;
			
			for(int i=0;i<n;i++)
				if(clr[i] == 'B')
				{
					if(a[i]<1)
						ok = false;
					else
						rgh.get(0).add(a[i]-1);
				}
				else
				{
					if(a[i]>n)
						ok = false;
					else
						rgh.get(Math.max(0, a[i]-1)).add(n-1);
				}
			
			
			TreeMap<Integer, Integer> q = new TreeMap<>();
			for(int i=0;i<n;i++)
			{
				for(int j: rgh.get(i))
					q.put(j, q.getOrDefault(j, 0)+1);
				if(q.isEmpty() || q.firstKey() < i)
				{
					ok = false;
					break;
				}
				Entry<Integer, Integer> cur = q.firstEntry();
				
				if(cur.getValue() == 1)
					q.remove(cur.getKey());
				else
					q.put(cur.getKey(), cur.getValue()-1);
			}
			out.println(ok? "YES":"NO");
			out.flush();
		}
	
	}

}

