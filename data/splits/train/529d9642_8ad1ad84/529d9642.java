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
public class C {

	private static ArrayList<Integer>[] adj;
	private static ArrayList<Pair> lis;
	private static Pair[] arr;
	

	public static void process() throws IOException {

		int n = sc.nextInt();
		lis = new ArrayList<C.Pair>();
		for(int i = 0; i<n; i++)lis.add(new Pair(sc.nextInt(), sc.nextInt()));
		adj = new ArrayList[n+1];
		for(int i = 0; i<=n; i++)adj[i] = new ArrayList<Integer>();
		for(int i = 1; i<n; i++) {
			int u = sc.nextInt(),v = sc.nextInt();
			adj[u].add(v);
			adj[v].add(u);
		}
		arr = new Pair[n+1];
		for(int i = 1; i<=n; i++)arr[i] = new Pair(0, 0);
		dfs(1,-1);
		System.out.println(Math.max(arr[1].x,arr[1].y));
	}

	

	private static void dfs(int node, int par) {
		for(int child : adj[node]) {
			if(child == par)continue;
			dfs(child, node);
			long ff = lis.get(node-1).x;
			long ss = lis.get(node-1).y;
			long cff = lis.get(child-1).x;
			long css = lis.get(child-1).y;
			
			
			// first 
			long val1 = Math.abs(ff-cff) + arr[child].x;
			long val2 = Math.abs(ff-css) + arr[child].y;
			if(val1 > val2) {
				arr[node].x+=val1;
			}
			else {
				arr[node].x+=val2;
			}
			// second 
			val1 = Math.abs(ss-cff) + arr[child].x;
			val2 = Math.abs(ss-css) + arr[child].y;
			if(val1 > val2) {
				arr[node].y+=val1;
			}
			else {
				arr[node].y+=val2;
			}
		}
		
		
	}



	



	//=============================================================================
	//--------------------------The End---------------------------------
	//=============================================================================

	

	static FastScanner sc;
	static PrintWriter out;

	public static void main(String[] args) throws IOException {
		boolean oj = true;
		if (oj) {
			sc = new FastScanner();
			out = new PrintWriter(System.out);
		} else {
			sc = new FastScanner(100);
			out = new PrintWriter("output.txt");
		}
		int t = 1;
		t = sc.nextInt();
		int TTT = 1;
		while (t-- > 0) {
			//			google(TTT++);
			process();
		}
		out.flush();
		out.close();
	}

	static class Pair implements Comparable<Pair> {
		long x, y;

		Pair(long x, long y) {
			this.x = x;
			this.y = y;
		}

		

		//		 @Override
		//		    public boolean equals(Object o) {
		//		        if (this == o) return true;
		//		        if (!(o instanceof Pair)) return false;
		//		        Pair key = (Pair) o;
		//		        return x == key.x && y == key.y;
		//		    }
		//		 
		//		    @Override
		//		    public int hashCode() {
		//		        int result = x;
		//		        result = 31 * result + y;
		//		        return result;
		//		    }
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////

	

	

	


	

	

	

	

	

	

	

	/////////////////////////////////////////////////////////////////////////////////////////////////////////

	

	

	
}
