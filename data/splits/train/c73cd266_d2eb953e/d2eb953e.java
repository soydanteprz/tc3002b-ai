import java.util.*;
// import java.lang.*;
import java.io.*;

//           THIS TEMPLATE MADE BY AKSH BANSAL.

public class Solution {
    
    private static boolean[] isPrime;
    
    
    
    static ArrayList<Integer>[] adj;
    static void getAdj(int n,int q, FastReader sc){
        adj = new ArrayList[n+1];
        for(int i=1;i<=n;i++){
            adj[i] = new ArrayList<>();
        }
        for(int i=0;i<q;i++){
            int a = sc.nextInt();
            int b = sc.nextInt();

            adj[a].add(b);
            adj[b].add(a);
        }
    }
    static PrintWriter out;
    public static void main(String[] args) throws IOException {
        FastReader sc = new FastReader();
        out = new PrintWriter(System.out);
        // primes();
        // ________________________________

        int t = sc.nextInt();
        StringBuilder output = new StringBuilder();

        while (t-- > 0) {
            int n = sc.nextInt();
            pair = new int[n+1][2];
            dp = new long[n+1][2];
            for(int i=1;i<=n;i++){
                pair[i][0]=sc.nextInt();
                pair[i][1]=sc.nextInt();
            }
            getAdj(n, n-1, sc);
            output.append(solver(n)).append("\n");
        }

        out.println(output);
        // _______________________________

        // int n = sc.nextInt();
        // out.println(solver());
        // ________________________________
        out.flush();
    }

    static int[][] pair;
    static long[][] dp;

    public static long solver(int n) {
        dfs(1,  0);
        return Math.max(dp[1][0], dp[1][1]);
    }

    static void dfs(int node, int parent){
        for(Integer child: adj[node]){
            if(child!=parent){
                dfs(child,  node);
                long left1 = Math.abs(pair[node][0]-pair[child][0]) + dp[child][0];
                long right1 = Math.abs(pair[node][0]-pair[child][1]) + dp[child][1];
                long left2 = Math.abs(pair[node][1]-pair[child][0]) + dp[child][0];
                long right2 = Math.abs(pair[node][1]-pair[child][1]) + dp[child][1];
                dp[node][0] += Math.max(left1, right1);
                dp[node][1] += Math.max(left2, right2);
            }
        }
    }
}
