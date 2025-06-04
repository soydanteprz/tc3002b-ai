import static java.lang.Math.max;
import static java.lang.Math.min;
import static java.lang.Math.abs;
import java.util.*;
import java.io.*;
import java.math.*;

/**
 *
 * @Har_Har_Mahadev
 */

/**
 * Main , Solution , Remove Public
 */
public class E {

	public static void process() throws IOException {

		int n = sc.nextInt(),k = sc.nextInt();
		int index[] = sc.readArray(k);
		long t[] = sc.readArrayLong(k);
		PriorityQueue<Pair> q = new PriorityQueue<Pair>();
		for(int i = 0; i<k; i++)q.add(new Pair(t[i]+index[i], index[i]));
		long ans[] = new long[n+1];
		Arrays.fill(ans, INF);
		int i = 1;
		while(!q.isEmpty() && i<=n) {
			while(!q.isEmpty() && q.peek().y < i) {
				q.poll();
			}
			if(q.isEmpty())break;
			Pair e = q.peek();
			ans[i]=min(ans[i],e.x-i);
			i++;
		}
//		System.out.println(Arrays.toString(ans));
		q.clear();
		for(i = 0; i<k; i++) {
			q.add(new Pair(t[i]+n-index[i], index[i]));
		}
		i = n;
		while(!q.isEmpty() && i>=1) {
			while(!q.isEmpty() && q.peek().y > i) {
				q.poll();
			}
			if(q.isEmpty())break;
			Pair e = q.peek();
			ans[i]=min(ans[i],e.x-(n-i));
			i--;
		}
		for( i = 1; i<=n; i++)out.print(ans[i]+" ");
		out.println();

	}

	//=============================================================================
	//--------------------------The End---------------------------------
	//=============================================================================
	private static long INF = 2000000000000000000L, M = 1000000007, MM = 998244353;
	private static int N = 0;

	

	static FastScanner sc;
	static FastWriter out;

	public static void main(String[] args) throws IOException {
		boolean oj = true;
		if (oj) {
			sc = new FastScanner();
			out = new FastWriter(System.out);
		} else {
			sc = new FastScanner("input.txt");
			out = new FastWriter("output.txt");
		}
		long s = System.currentTimeMillis();
		int t = 1;
		t = sc.nextInt();
		int TTT = 1;
		while (t-- > 0) {
			//			google(TTT++);
			process();
		}
		out.flush();
		//		tr(System.currentTimeMillis()-s+"ms");
	}

	private static boolean oj = System.getProperty("ONLINE_JUDGE") != null;

	

	static class Pair implements Comparable<Pair> {
		long x;
		int y;

		Pair(long x, int y) {
			this.x = x;
			this.y = y;
		}

		

		/*
		 	@Override
		    public boolean equals(Object o) {
		        if (this == o) return true;
		        if (!(o instanceof Pair)) return false;
		        Pair key = (Pair) o;
		        return x == key.x && y == key.y;
		    }
		 
		    @Override
		    public int hashCode() {
		        int result = x;
		        result = 31 * result + y;
		        return result;
		    }
		*/
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////

	

	

	

	

	

	

	

	

	

	

	

	

	//custom multiset (replace with HashMap if needed)
	

	

	// compress Big value to Time Limit
	

	// Fast Writer 

	

	// Fast Inputs
	

}
