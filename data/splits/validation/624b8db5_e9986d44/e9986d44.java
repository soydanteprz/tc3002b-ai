import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;



public class cp23 {
	static BufferedReader sc = new BufferedReader(new InputStreamReader(System.in));
	static int mod = 1000000007;
	static String toReturn = "";

	static int steps = Integer.MAX_VALUE;
	static int maxlen = 1000005;
	
	
	
	/*MATHEMATICS FUNCTIONS START HERE
	MATHS
	MATHS
	MATHS
	MATHS*/
	

	
	
	
	
	
	/*
	 * static long [] seive(long n) {
	 * 
	 * }
	 */
	/*MATHS
	MATHS
	MATHS
	MATHS
	MATHEMATICS FUNCTIONS END HERE */
	
	/*SWAP FUNCTION START HERE
	  SWAP
	  SWAP
	  SWAP
	  SWAP
	 */
	
	/*SWAP
	  SWAP
	  SWAP
	  SWAP
	 SWAP FUNCTION END HERE*/
	
	
	
	/*BINARY SEARCH METHODS START HERE
	 * BINARY SEARCH
	 * BINARY SEARCH
	 * BINARY SEARCH
	 * BINARY SEARCH
	 */
	
	
	
	/*BINARY SEARCH
	 * BINARY SEARCH
	 * BINARY SEARCH
	 * BINARY SEARCH
	 * BINARY SEARCH
	 BINARY SEARCH METHODS END HERE*/
	
	
	/*RECURSIVE FUNCTION START HERE
	 * RECURSIVE
	 * RECURSIVE
	 * RECURSIVE
	 * RECURSIVE
	 */
	
	
	/*RECURSIVE
	 * RECURSIVE
	 * RECURSIVE
	 * RECURSIVE
	 * RECURSIVE
	 RECURSIVE FUNCTION END HERE*/
	
	/*GRAPH FUNCTIONS START HERE
	 * GRAPH
	 * GRAPH
	 * GRAPH
	 * GRAPH
	 * */
	static class edge{
		int from, to, weight;
		public edge(int x, int y, int z) {
			this.from = x;
			this.to = y;
			this.weight = z;
		}
	}
	static void addEdge(ArrayList<ArrayList<edge>> graph, int from, int to, int weight) {
		edge temp = new edge(from, to, weight);
		edge temp1 = new edge(to, from, weight);
		graph.get(from).add(temp);
		graph.get(to).add(temp1);
	}
	static int ans = 0;
	static void dfs(ArrayList<ArrayList<edge>> graph, int vertex, boolean [] visited, int [] toReturn, int weight) {
		//System.out.println(graph.get(vertex).size());
		if(visited[vertex]) return;
		visited[vertex] = true;
		if(graph.get(vertex).size() > 2) return;
		for(int i = 0; i < graph.get(vertex).size(); i++) {
			edge temp = graph.get(vertex).get(i);
			if(!visited[temp.to]) {
				//System.out.println(temp.to);
				toReturn[temp.weight] = weight;
				dfs(graph, temp.to, visited, toReturn, 5 - weight);
				weight = 5 - weight;
			}
		}
	}
	
	
	
	
	/*GRAPH FUNCTIONS END HERE
	 * GRAPH
	 * GRAPH
	 * GRAPH
	 * GRAPH
	 */
	/*disjoint Set START HERE
	 * disjoint Set
	 * disjoint Set
	 * disjoint Set
	 * disjoint Set
	 */
	static int [] rank;
	static int [] parent;
	
	
	/*disjoint Set END HERE
	 * disjoint Set
	 * disjoint Set
	 * disjoint Set
	 * disjoint Set
	 */
	
	
	/*INPUT START HERE
	 * INPUT
	 * INPUT
	 * INPUT
	 * INPUT
	 * INPUT
	 */
	
	static int nexInt() throws NumberFormatException, IOException {
		return Integer.parseInt(sc.readLine());
	}
	
	
	static int [] inputIntArr() throws NumberFormatException, IOException{
		String [] s = sc.readLine().split(" ");
		int [] toReturn = new int[s.length];
		for(int i = 0; i < s.length; i++) {
			toReturn[i] = Integer.parseInt(s[i]);
		}
		return toReturn;
	}
	/*INPUT
	 * INPUT
	 * INPUT
	 * INPUT
	 * INPUT
	 * INPUT END HERE
	 */
	static void solve() throws IOException {
		
		int n = nexInt();
		ArrayList<ArrayList<edge>> tree = new ArrayList<>();
		for(int i = 0; i < n; i++) tree.add(new ArrayList<edge>());
		for(int i = 0; i < n - 1; i++) {
		int [] s1 = inputIntArr();
		addEdge(tree, s1[0] - 1, s1[1] - 1, i);
		}
		int vertex = 0;
		for(int i = 0; i < tree.size(); i++) {
			if(tree.get(i).size() > 2) {
				System.out.println(-1);
				return;
			}else if(tree.size() == 1) {
				vertex = i;
			}
		}
		int [] toReturn = new int[n - 1];
		//toReturn.add(2);
		dfs(tree, vertex, new boolean[n], toReturn,  2);
		//System.out.println(toReturn.size());
		
		
			for(int i = 0; i <  toReturn.length; i++) 
				if(toReturn[i] != 0)System.out.print(toReturn[i] + " ");
			System.out.println();
		
		
	}

	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub
		
		 int t = Integer.parseInt(sc.readLine()); for(int i = 0; i < t; i++) 
		  solve();
		
		
	}
	}
