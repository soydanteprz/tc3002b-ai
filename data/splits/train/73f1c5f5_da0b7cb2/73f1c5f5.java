import java.util.*;
import java.io.*;

public class Ana {
	static int n, a, b, da, db;
	static ArrayList<Integer>[] graph;

	public static void main(String[] args) throws IOException, InterruptedException {

		Scanner sc = new Scanner(System.in);
		PrintWriter out = new PrintWriter(System.out);
		int t = sc.nextInt();
		while (t-- > 0) {
			n = sc.nextInt();
			a = sc.nextInt() - 1;
			b = sc.nextInt() - 1;
			da = sc.nextInt();
			db = sc.nextInt();
			graph = new ArrayList[n];
			for (int i = 0; i < n; i++)
				graph[i] = new ArrayList<Integer>();
			for (int i = 0; i < n - 1; i++) {
				int u = sc.nextInt() - 1, v = sc.nextInt() - 1;
				graph[u].add(v);
				graph[v].add(u);
			}
			int[] dist = new int[n];
			Arrays.fill(dist, -1);
			bfs(a, dist);
			boolean bobwins = true; // if false alice wins
			if (dist[b] <= da || 2 * da >= db)
				bobwins = false;
			int far = a;
			for (int i = 0; i < n; i++)
				if (dist[i] > dist[far])
					far = i;
			Arrays.fill(dist, -1);
			bfs(far, dist);
			for (int i = 0; i < n; i++)
				if (dist[i] > dist[far])
					far = i;
			if(2*da>=dist[far])//max path
				bobwins = false;
//			out.println(dist[far]);
			out.println(bobwins?"Bob":"Alice");
		}

		out.close();
	}

	static void bfs(int s, int[] dist) {
		Queue<Integer> q = new LinkedList<Integer>();
		q.add(s);
		dist[s] = 0;
		while (!q.isEmpty()) {
			int u = q.remove();
			for (int v : graph[u]) {
				if (dist[v] == -1) {
					dist[v] = 1 + dist[u];
					q.add(v);
				}
			}
		}
	}

	

}
