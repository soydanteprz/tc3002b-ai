import java.io.*;
import java.util.*;
		
public class practice {
	
	static FastScanner fs = new FastScanner();

	public static void main(String[] args) {
		int t = 1;
		t = fs.nextInt();   
		for(int i=1;i<=t;i++) {
			solve(t);
		}
	}
	
	public static void dfs(int u, boolean visited[], ArrayList<ArrayList<Integer>> g[], int ans[], int x) {
		visited[u] = true;
		for(ArrayList<Integer> v:g[u]) {
			if(!visited[v.get(0)]) {
				ans[v.get(1)]=x;
				dfs(v.get(0),visited,g,ans,x^1);
			}
		}
		return;
	}
	
	@SuppressWarnings("unused")
	public static void solve(int tt) {
		int n = fs.nextInt();
		@SuppressWarnings("unchecked")
		ArrayList<ArrayList<Integer>> g[] = new ArrayList[n];
		for(int i=0;i<n;i++)g[i]=new ArrayList<ArrayList<Integer>>();
		int ans[] = new int[n];
		boolean visited[] = new boolean[n];
		int deg[] = new int[n];
		for(int i=0;i<n;i++)deg[i]=0;
		for(int i=0;i<n-1;i++) {
			int u = fs.nextInt() - 1;int v = fs.nextInt() - 1;
			ArrayList<Integer> L1=new ArrayList<Integer>();
			L1.add(v);L1.add(i);
			g[u].add(L1);
			ArrayList<Integer> L2=new ArrayList<Integer>();
			L2.add(u);L2.add(i);
			g[v].add(L2);
			deg[u]++;
			deg[v]++;
		}
		// degree must be <=2 cause if greater than that then sum will be always non prime
		// there will be exactly 2 leafs, we start dfs from one of them
		int leaf=-1;
		boolean notcool=false;
		for(int i=0;i<n;i++) {
			if(deg[i]>2) {
				System.out.println(-1);
				return;
			}
		}
		for(int i=0;i<n;i++) {
			if(deg[i]==1) {
				leaf=i;
				break;
			}
		}
		dfs(leaf,visited,g,ans,2);
		for(int i=0;i<n-1;i++)System.out.print(ans[i]+" ");
		System.out.println();
		return;
	}
	
	
	
	
}
