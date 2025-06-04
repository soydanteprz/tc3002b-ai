import java.io.*;
import java.util.*;
import java.util.Map.Entry;

import static java.lang.Math.*;

public class Ans implements Runnable {

	public static void main(String args[]) {
		Ans s = new Ans();
		s.run();
	}

	

	InputReader sc = null;
	PrintWriter pw = null;
	static ArrayList<Integer>[] G = new ArrayList[(int)(3e5+10)];
	static int[] dist = new int[(int)(3e5+10)];
	static int[] a = new int[(int)(3e5+10)];
	
	private static int[] nge(int n){		
		int[] edges = new int[n];
		Arrays.fill(edges,  -1);
		Stack<Integer> st = new Stack<>();
		st.push(0);
		for(int i = 1; i < n; i++){			
			while(!st.isEmpty() && a[i] >= a[st.peek()]){
				edges[st.pop()] = i;
			}
			st.push(i);			
		}		
		//debug("nge", edges);
		return edges;
	}
	
	private static int[] nle(int n){	
		int[] edges = new int[n];
		Arrays.fill(edges,  -1);
		Stack<Integer> st = new Stack<>();
		st.push(0);
		for(int i = 1; i < n; i++){
			while(!st.isEmpty() && a[i] <= a[st.peek()]){
				edges[st.pop()] = i;
			}
			st.push(i);
		}
		//debug("nle", edges);
		return edges;
	}
	
	private static int[] pge(int n){		
		int[] edges = new int[n];
		Arrays.fill(edges,  -1);
		Stack<Integer> st = new Stack<>();
		st.push(n-1);
		for(int i = n-2; i >= 0; i--){
			while(!st.isEmpty() && a[i] >= a[st.peek()]){
				edges[st.pop()] = i;
			}
			st.push(i);
		}
		//debug("pge", edges);
		return edges;
	}
	
	private static int[] ple(int n){	
		int[] edges = new int[n];
		Arrays.fill(edges,  -1);
		Stack<Integer> st = new Stack<>();
		st.push(n-1);
		for(int i = n-2; i >= 0; i--){
			while(!st.isEmpty() && a[i] <= a[st.peek()]){
				edges[st.pop()] = i;
			}
			st.push(i);
		}
		//debug("ple", edges);
		return edges;
	}
	
	private static void buildGraph(int[] edges){		
		for(int i = 0; i < edges.length; i++){
			if(edges[i] != -1){				
//				G[i].add(edges[i]);
//				G[edges[i]].add(i);
				G[min(i, edges[i])].add(max(i, edges[i]));
			}
		}
	}
	
	private static void bfs(int n){		
		dist[0] = 0;
		ArrayDeque<Integer> q = new ArrayDeque<>();
		q.add(0);
		
		while(!q.isEmpty()){
			int front = q.pollFirst();
			if(front == n-1){
				break;
			}
			for(int adj : G[front]){
				if(dist[adj] == (int)(1e9)){
					dist[adj] = 1 + dist[front];
					q.add(adj);
				}
			}
		}
	}
	
	public void run() {

		// InputStream is;
		// is = new FileInputStream(new File("input.txt"));
		sc = new InputReader(System.in);
		pw = new PrintWriter(System.out);
		int n = sc.nextInt();
		a = new int[n];		
		for (int i = 0; i < n; i++) {
			a[i] = sc.nextInt();
			G[i] = new ArrayList<Integer>();
		}		
		
		buildGraph(nge(n));
		buildGraph(nle(n));
		buildGraph(ple(n));
		buildGraph(pge(n));
		Arrays.fill(dist, (int)(1e9));
		
		bfs(n);
		
		pw.println(dist[n-1]);
		
		// is.close();
		pw.close();

	}
		
	
}
