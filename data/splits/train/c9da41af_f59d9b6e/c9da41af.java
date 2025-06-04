import java.util.*;
import java.lang.*;
import java.io.*;



public class Main {
	static {
		try {
			System.setIn(new FileInputStream("input.txt"));
			System.setOut(new PrintStream(new FileOutputStream("output.txt")));
		} catch (Exception e) {}
	}
	void solve() {
		int n = in.nextInt();

		ArrayList<Edge>[] graph = new ArrayList[n + 1];
		for (int i = 0; i < n; i++) {
			graph[i] = new ArrayList<Edge>();
		}

		for (int i = 0; i < n - 1; i++) {
			int u = in.nextInt();
			int v = in.nextInt();
			v--; u--;
			graph[u].add(new Edge(v, i));
			graph[v].add(new Edge(u, i));
		}

		int[] res = new int[n - 1];

		for (int i = 0; i < n; i++) {
			if (graph[i].size() > 2) {
				out.append("-1\n");
				return;
			}
		}

		int start = -1;
		for (int i = 0; i < n; i++) {
			if (graph[i].size() == 1) {
				start  = i;
				break;
			}
		}

		int currNode = start;
		int prevNode = -1;
		int weight = 2;

		while (true) {
			ArrayList<Edge> edges = graph[currNode];
			Edge next = edges.get(0);
			if (next.node == prevNode) {
				if (edges.size() == 1) {
					break;
				}
				next = edges.get(1);
			}
			res[next.index] = weight;
			weight = 5 - weight;
			prevNode = currNode;
			currNode = next.node;
		}

		for (int i = 0; i < n - 1; i++) {
			out.append(res[i] + " ");
		}
		out.append("\n");
	}

	public static void main (String[] args) {
		// Its Not Over Untill I Win - Syed Mizbahuddin
		Main sol = new Main();
		int t = 1;
		t = in.nextInt();
		while (t-- != 0) {
			sol.solve();
		}
		System.out.print(out);
	}

	

	

	void print(int s) {
		System.out.print(s);
	}

	

	

	

	


	static FastReader in;
	static StringBuffer out;
	final int MAX;
	final int MIN;
	int mod ;
	Main() {
		in = new FastReader();
		out = new StringBuffer();
		MAX = Integer.MAX_VALUE;
		MIN = Integer.MIN_VALUE;
		mod = (int)1e9 + 7;
	}
}


class Edge {
	int node, index;
	Edge(int node, int index) {
		this.node = node;
		this.index = index;
	}
}
