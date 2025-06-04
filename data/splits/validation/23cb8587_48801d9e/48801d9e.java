import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.*;

public class C {
    private static FastReader fr = new FastReader();
    private static PrintWriter out=new PrintWriter(System.out);
    private static Random random = new Random();

    private static long[][] dp;

    private static long calculate(List<Integer>[] graph, int current, long[][] r, boolean[] stack, int use){
        if(dp[current][use] != -1) return dp[current][use];
        stack[current] = true;
        long max = 0;
        if(graph[current] != null){
            for(int next : graph[current]){
                if(!stack[next]){
                    stack[next] = true;
                    long r1 = Math.abs(r[current][use] - r[next][0]) + calculate(graph, next, r, stack, 0);
                    long r2 = Math.abs(r[current][use] - r[next][1]) + calculate(graph, next, r, stack, 1);
                    max += Math.max(r1, r2);
                }
            }
        }
        stack[current] = false;
        dp[current][use] = max;
        return max;
    }

    public static void main(String[] args) throws IOException {
        StringBuilder sb = new StringBuilder();
        // code goes here
        int t = fr.nextInt();
        while (t-- > 0){
            int n = fr.nextInt();
            long[][] r = new long[n][2];
            for(int i = 0; i < n; i++){
                r[i] = fr.nextLongArray(2);
            }
            List<Integer>[] graph = new ArrayList[n];
            for(int i = 0; i < n - 1; i++){
                int u = fr.nextInt();
                int v = fr.nextInt();
                if(graph[u - 1] == null) graph[u - 1] = new ArrayList<>();
                if(graph[v - 1] == null) graph[v - 1] = new ArrayList<>();
                graph[u - 1].add(v - 1);
                graph[v - 1].add(u - 1);
            }

            boolean[] stack = new boolean[n];
            dp = new long[n][2];
            for(int i = 0; i < dp.length; i++){
                Arrays.fill(dp[i], -1);
            }

            long r1 = calculate(graph, 0, r, stack, 0);
            long r2 = calculate(graph, 0, r, stack, 1);
            sb.append(Math.max(r1, r2)).append("\n");
        }
        System.out.print(sb.toString());
    }

    

    

    

    

    

    

    // Returns n^(-1) mod p
    

    // Returns nCr % p using Fermat's
    // little theorem.
    
}
