import java.io.*;
import java.text.DecimalFormat;
import java.util.*;

public class C {
    static long mod = (long) 1e9 + 7;
    static long mod1 = 998244353;
    static ArrayList<Integer>[] adj;
    static HashMap<String,Long> hmap;
    public static void main(String[] args) {
        InputStream inputStream = System.in;
        OutputStream outputStream = System.out;
        InputReader in = new InputReader(inputStream);
        PrintWriter out = new PrintWriter(outputStream);


        int t = in.nextInt();
        while (t-- > 0) {

            int n = in.nextInt();
            adj = new ArrayList[n + 1];
            for (int i = 0; i <= n; i++)
                adj[i] = new ArrayList<>();
            long[][] dp = new long[n + 1][2];
            hmap=new HashMap<>();
            for (int i = 1; i <= n; i++) {
                dp[i][0] = in.nextLong();
                dp[i][1] = in.nextLong();
            }
            for (int i = 0; i < n - 1; i++) {
                int u = in.nextInt();
                int v = in.nextInt();
                adj[u].add(v);
                adj[v].add(u);
            }
            long[] x=DFS(dp,1,-1);
            out.println(Math.max(x[0],x[1]));
        }
        out.close();
    }

    static long[] DFS(long[][] dp,int s,int prev) {

        long max=0;
        long min=0;
        for (int i : adj[s]) {
            if (i != prev) {
                long[] dd=DFS(dp,i,s);
                min+=Math.max(Math.abs(dp[s][0]-dp[i][0])+dd[0],Math.abs(dp[s][0]-dp[i][1])+dd[1]);
                max+=Math.max(Math.abs(dp[s][1]-dp[i][0])+dd[0],Math.abs(dp[s][1]-dp[i][1])+dd[1]);
            }
        }
        return new long[] {min,max};

    }


    static final Random random = new Random();

    

    

    

    

    

    

    

    

    
}
