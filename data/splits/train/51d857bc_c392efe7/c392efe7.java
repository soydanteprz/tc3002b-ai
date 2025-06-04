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
public class D {

	static int visited[];
	static int distance[];
	private static ArrayList<Integer>[] adj;
	 
	
	private static void BFS(int node) {
		
		visited[node] = 1;
		Queue<Integer> q = new LinkedList<Integer>();
		q.offer(node);
		distance[node] = 0;
		
		while(!q.isEmpty()) {
			int curr = q.poll();
			for(int child : adj[curr]) {
				if(visited[child] == 0) {
					q.offer(child);
					distance[child] = distance[curr] + 1;
					visited[child] = 1;
				}
			}
		}
		
	}
	

	public static void process() throws IOException {

		int n = sc.nextInt(),a = sc.nextInt(),b = sc.nextInt(),
				da = sc.nextInt(),db = sc.nextInt();
		adj = new ArrayList[n+1];
		distance = new int[n+1];
		visited = new int[n+1];
		for(int i = 0; i<=n; i++)adj[i] = new ArrayList<Integer>();
		for(int i =1; i<n; i++) {
			int u = sc.nextInt(),v = sc.nextInt();
			adj[u].add(v);
			adj[v].add(u);
		}
		BFS(a);
		if(distance[b] <= da || db-da<=da) {
			System.out.println("Alice");
			return;
		}
		ArrayList<Pair> lis = new ArrayList<D.Pair>();
		for(int i = 1; i<=n; i++)lis.add(new Pair(distance[i], i));
		Collections.sort(lis);
		Pair e = lis.get(n-1);
		distance = new int[n+1];
		visited = new int[n+1];
		BFS(e.y);
		int max = 0;
		for(int i = 1; i<=n; i++)max = Math.max(max,distance[i]);
		if(max-da<=da) {
			System.out.println("Alice");
			return;
		}
		System.out.println("Bob");
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
		int x, y;

		Pair(int x, int y) {
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
