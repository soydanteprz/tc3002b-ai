import java.util.*;
import java.io.*;
import java.math.*;

public class C1529 {
    public static void main(String[] args) throws IOException {
        PrintWriter out = new PrintWriter(System.out);
        Reader s = new Reader();
        int t = s.i();
        while (t-- > 0) {
            int n = s.i();
            pairLong [] arr = new pairLong[n];
            for (int i=0;i<n;i++) {
                long x = s.l() , y = s.l();
                arr[i] = new pairLong(x,y);
            }
            ArrayList<Integer>[] tree = new ArrayList[n+1];
            for (int i=0;i<=n;i++) tree[i] = new ArrayList<>();
            for (int i=0;i<n-1;i++) {
                int x = s.i() , y = s.i();
                tree[x].add(y);
                tree[y].add(x);
            }
            pairLong[] dp = new pairLong[n+1];
            boolean [] vis = new boolean[n+1];
            pairLong x = value(tree,arr,dp,1,vis);
            out.println(Math.max(x.first,x.second));
        }
        out.flush();
    }
    private static pairLong value(ArrayList<Integer>[] tree , pairLong [] arr , pairLong [] dp , int in , boolean [] vis) {
        if (dp[in] != null) return dp[in];
        vis[in] = true;
        long a = 0L , b = 0L;
        for (int x : tree[in]) {
            if (!vis[x]) {
                pairLong y = value(tree,arr,dp,x,vis);
                a += Math.max(Math.abs(arr[in-1].first-arr[x-1].first) + y.first,Math.abs(arr[in-1].first-arr[x-1].second) + y.second);
                b += Math.max(Math.abs(arr[in-1].second-arr[x-1].first) + y.first,Math.abs(arr[in-1].second-arr[x-1].second) + y.second);
            }
        }
        dp[in] = new pairLong(a,b);
        return dp[in];
    }
    
    

    

    

    

    

    

    

    

    

    
    static class pairLong implements Comparator<pairLong> {
        long first, second;

        pairLong() {
        }

        pairLong(long first, long second) {
            this.first = first;
            this.second = second;
        }

        
    }
//    static class pair implements Comparator<pair> {
//        int first, second;
//
//        pair() {
//        }
//
//        pair(int first, int second) {
//            this.first = first;
//            this.second = second;
//        }
//
//        @Override
//        public int compare(pair p1, pair p2) {
//            if (p1.first == p2.first) return p1.second - p2.second;
//            return p1.first - p2.first;
//        }
//    }
}
