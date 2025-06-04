import java.io.*;
import java.util.*;

public class Codeforces {

    


    static FastReader f = new FastReader();
    static BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
    static StringTokenizer st;
    static StringBuilder sb = new StringBuilder("");
    private static int m = (int) 1e9 + 7;

    static int[] inputArray(int n) throws IOException {
        int[] a = new int[n];
        for (int i = 0; i < n; i++) {
            a[i] = f.nextInt();
        }
        return a;
    }

    

    

    

    public static void main(String[] args) throws IOException {

//        System.out.println(gcd(11, 3));

        int t = 1;
        while(t-- != 0) {
            int N = f.nextInt();
            int a[] = inputArray(N);
            ArrayList<Integer> space = new ArrayList<>();
            ArrayList<Integer> fill = new ArrayList<>();
            for(int i = 0 ; i < N;  i++) {
                if(a[i] == 0) {
                    space.add(i);
                }
                else {
                    fill.add(i);
                }
            }
            int n = fill.size(), m = space.size();
            int[][] dp = new int[n+1][m+1];
            // i - no of filled
            // j is no of space
            Arrays.fill(dp[0], 0);
            for(int i = 1 ; i <= n ; i++) {
                dp[i][0] = -1;
            }
            for(int i = 1 ; i <= n ; i++) {
                for(int j = 1 ; j <= m ; j++) {
                    if(i > j) {
                        dp[i][j] = -1;
                        continue;
                    }
                    dp[i][j] = Integer.MAX_VALUE;
                    if(dp[i-1][j-1] != -1) {
                        dp[i][j] = Math.min(dp[i][j], dp[i-1][j-1]) + Math.abs(fill.get(i-1) - space.get(j-1));
                    }
                    if(dp[i][j-1] != -1) {
                        dp[i][j] = Math.min(dp[i][j], dp[i][j-1]);
                    }
                }
            }

            System.out.println(dp[n][m]);

        }

        System.out.println(sb);

    }

}
