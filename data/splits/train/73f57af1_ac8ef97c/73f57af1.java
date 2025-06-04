import java.util.*;
import java.lang.*;
import java.io.*;

public class Main
{
	static PrintWriter out;
	static FastReader scan;

    /*-------- I/O usaing short named function ---------*/
	
	public static int ni(){return scan.nextInt();}
	public static long nl(){return scan.nextLong();}
	
	
	
	
	public static void pn(Object o){out.println(o);}
    /*-------- for output of an array ---------------------*/
     
	
	
	
    /*-------------- for input in an array ---------------------*/
	
	
	
	
	/*------------ for taking input faster ----------------*/
     

	// Method to check if x is power of 2
    
    //Method to return lcm of two numbers
    
    //Method to count digit of a number
    

    //Method for sorting
	
	//Method for checking if a number is prime or not
	 

    static long[] l, r;
	public static void main (String[] args) throws java.lang.Exception
	{
		OutputStream outputStream =System.out;
        out =new PrintWriter(outputStream);
        scan =new FastReader();
		//for fast output sometimes
		StringBuilder sb = new StringBuilder();
		int t = ni();
		while(t-->0){
			int n = ni();
			l = new long[n];
			r = new long[n];

			for(int i=0; i<n; i++){
				l[i] = nl();
				r[i] = nl();
			}

			//lPA(l);
			//lPA(r);

			ArrayList<Integer> adj[] = new ArrayList[n];
			for(int i=0; i<n; i++)
				adj[i] = new ArrayList<Integer>();

			for(int i=0; i<n-1; i++){
				int u = ni()-1, v = ni()-1;
				adj[u].add(v);
				adj[v].add(u);
			}

			dp = new Long[n][2];
			visited = new boolean[n];
			long ans = Math.max(solve(adj, 0, 0, visited), solve(adj, 0, 1, visited));
			pn(ans);
        }
        out.flush();
		out.close();
	}

	static Long dp[][];
	static boolean visited[];
	static long solve(ArrayList<Integer> adj[], int vertex, int prev, boolean visited[]){

		visited[vertex] = true;
		if(dp[vertex][prev] != null)
			return dp[vertex][prev];
		long ans = 0;
		for(int x : adj[vertex]){
			if(!visited[x]){
				if(prev == 0){
					ans += Math.max(Math.abs(l[vertex] - l[x]) + solve(adj, x, 0, visited),
						Math.abs(l[vertex] - r[x]) + solve(adj, x, 1, visited));
					//pn(vertex + " " + x + " " + ans);
				}else{
					ans += Math.max(Math.abs(r[vertex] - l[x]) + solve(adj, x, 0, visited),
						Math.abs(r[vertex] - r[x]) + solve(adj, x, 1, visited));
					//pn(vertex + " " + x + " " + ans);
				}
			}
		}
		visited[vertex] = false;
		//pn(ans);
		return dp[vertex][prev] = ans;	
	}
	
	
}
