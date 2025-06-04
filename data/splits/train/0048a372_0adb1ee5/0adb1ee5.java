import java.util.*;
// import java.lang.*;
import java.io.*;

//           THIS TEMPLATE MADE BY AKSH BANSAL.

public class Solution {
    
    private static boolean[] isPrime;
    
    
    
    
    
    // static ArrayList<Integer>[] adj;
    // static void getAdj(int n,int q, FastReader sc){
    //     adj = new ArrayList[n+1];
    //     for(int i=1;i<=n;i++){
    //         adj[i] = new ArrayList<>();
    //     }
    //     for(int i=0;i<q;i++){
    //         int a = sc.nextInt();
    //         int b = sc.nextInt();

    //         adj[a].add(b);
    //         adj[b].add(a);
    //     }
    // }
    static PrintWriter out;
    static FastReader sc ;
    public static void main(String[] args) throws IOException {
        sc = new FastReader();
        out = new PrintWriter(System.out);
        // primes();
        // ________________________________

        int t = sc.nextInt();
        StringBuilder output = new StringBuilder();

        while (t-- > 0) {
            int n= sc.nextInt();
            int k= sc.nextInt();
            long[][] arr = new long[k][2];
            for(int i=0;i<k;i++){
                arr[i][0] = sc.nextLong();
            }
            for(int i=0;i<k;i++){
                arr[i][1] = sc.nextLong();
            }
            // output.append(solver()).append("\n");
            solver(arr, k, n);
        }

        out.println(output);
        // _______________________________

        // int n = sc.nextInt();
        // out.println(solver());
        // ________________________________
        out.flush();
    }

    public static void solver(long[][] arr, int k, int n) {
        HashMap<Long, Long> map = new HashMap<>();
        for(int i=0;i<k;i++){
            map.put(arr[i][0]-1, arr[i][1]);
        }

        long cur = Integer.MAX_VALUE;
        long[] left = new long[n];
        long[] right = new long[n];

        for(int i=0;i<n;i++){
            long temp = Integer.MAX_VALUE;
            if(map.containsKey((long)i)){
                temp = map.get((long)i);
            }
            cur = Math.min(cur+1, temp);
            left[i] = cur;
            // System.out.println("__"+ temp);
        }

        cur = Integer.MAX_VALUE;

        for(int i=n-1;i>=0;i--){
            long temp = Integer.MAX_VALUE;
            if(map.containsKey((long)i)){
                temp = map.get((long)i);
            }
            cur = Math.min(cur+1, temp);
            right[i] = cur;
        }
        for(int i=0;i<n;i++){
            out.print(Math.min(left[i], right[i])+" ");
        }
        out.println();
    }
}
