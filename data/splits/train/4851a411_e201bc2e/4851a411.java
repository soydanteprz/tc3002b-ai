import java.util.ArrayList;
import java.util.HashMap;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.StringTokenizer;
import java.util.Arrays;
import java.util.Random;
import java.io.FileWriter;
import java.io.PrintWriter;
/*
    Solution Created: 17:15:47 15/01/2022
    Custom Competitive programming helper.
*/

public class Main {
	static ArrayList<Integer> adj[];
	static HashMap<Integer, HashMap<Integer, Integer>> mp;
	public static void insert(int i, int j, int v) {
		int a = Math.min(i, j), b = Math.max(i, j);
		if(!mp.containsKey(a)) mp.put(a, new HashMap<>());
		mp.get(a).put(b, v);
	}
	public static int get(int i, int j) {
		int a = Math.min(i, j), b = Math.max(i, j);
		return mp.get(a).get(b);
	}
	public static void dfs(int i, int p, boolean two) {
		
		for(int to : adj[i]) {
			if(to == p) continue;
			insert(i, to, two?2:3);
			dfs(to,i,!two);
			two = !two;
		}
	}
	public static void solve() {
		mp = new HashMap<>();
		int n = in.nextInt();
		adj = new ArrayList[n];
		for(int i = 0; i<n; i++) adj[i] = new ArrayList<>();
		int[][] e = new int[n-1][2];
		int[] deg = new int[n];
		for(int i = 0; i<n-1; i++) {
			e[i] = in.na(2);
			deg[--e[i][0]]++;
			deg[--e[i][1]]++;
			adj[e[i][0]].add(e[i][1]);
			adj[e[i][1]].add(e[i][0]);
		}
		for(int i = 0; i<n; i++) {
			if(deg[i]>=3) {
				out.println(-1);
				return;
			}
		}
		dfs(0,-1,true);
		for(int i = 0; i<n-1; i++) {
			out.print(get(e[i][0], e[i][1])+" ");
		}
		out.println();
	}
	
	public static void main(String[] args) {
		in = new Reader();
		out = new Writer();
		int t = in.nextInt();
		while(t-->0) solve();
		out.exit();
	}
	static Reader in; static Writer out;






}
