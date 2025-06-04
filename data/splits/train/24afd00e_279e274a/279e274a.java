import java.io.*;
import java.util.*;

public class C {
    

    static FastReader s = new FastReader();
    static PrintWriter out = new PrintWriter(System.out);

    

    

    

    

    private static int ri() {
        return s.nextInt();
    }

    private static long rl() {
        return s.nextLong();
    }

    

    

    

    
    static int MOD=1000000007;

    

    static long[][] dp;
    static void dfs(List<Integer>[] list, long[] l, long[] r, boolean[] vis,int node)
    {
        vis[node] = true;

        for(int i:list[node])
        {
            if(!vis[i])
            {
                dfs(list,l,r,vis, i);
                dp[node][0] += Math.max(dp[i][0]+Math.abs(l[node]-l[i]), dp[i][1]+Math.abs(l[node]-r[i]));
                dp[node][1] += Math.max(dp[i][0]+Math.abs(r[node]-l[i]), dp[i][1]+Math.abs(r[node]-r[i]));
            }

        }
    }
    public static void main(String[] args) {
        StringBuilder ans = new StringBuilder();
        int t = ri();
//        int t = 1;
        while (t-- > 0)
        {
            int n=ri();
            dp = new long[n][2];
            long[] l = new long[n];
            long[] r = new long[n];
            for(int i=0;i<n;i++)
            {
                l[i] = rl();
                r[i] = rl();

            }

            List<Integer>[] list=new ArrayList[n];
            for(int i=0;i<n;i++)
            {
                list[i] = new ArrayList<>();
            }

            for(int i=0;i<n-1;i++)
            {
                int a= ri();int b=ri();
                a--;b--;
                list[a].add(b);
                list[b].add(a);
            }

            boolean[] vis = new boolean[n];
            dfs(list,l,r,vis,0);

//            for(int i=0;i<n;i++)
//            {
//                System.out.println(Arrays.toString(dp[i]));
//            }

            ans.append(Math.max(dp[0][0],dp[0][1])).append("\n");

        }
        out.print(ans.toString());
        out.flush();
    }



}
