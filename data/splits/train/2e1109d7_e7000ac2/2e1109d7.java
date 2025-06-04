import java.io.OutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.InputMismatchException;
import java.io.IOException;
import java.util.ArrayList;
import java.io.InputStream;


public class Main {
    public static void main(String[] args) {
        InputStream inputStream = System.in;
        OutputStream outputStream = System.out;
        FastReader in = new FastReader(inputStream);
        PrintWriter out = new PrintWriter(outputStream);
        DArmchairs solver = new DArmchairs();
        solver.solve(1, in, out);
        out.close();
    }

    static class DArmchairs {
        ArrayList<Integer>[] arr;
        long[][] dp;

        public void solve(int testNumber, FastReader in, PrintWriter out) {
            int n = in.nextInt();
            int[] a = in.readArray(n);

            int mx = 5001;
            dp = new long[mx][mx];
            for (int i = 0; i < mx; ++i) {
                Arrays.fill(dp[i], -1);
            }

            arr = new ArrayList[2];
            for (int i = 0; i < 2; ++i) {
                arr[i] = new ArrayList<>();
            }

            for (int i = 0; i < n; ++i) {
                arr[a[i]].add(i);
            }

            out.println(go(0, 0));
        }

        long go(int i, int j) {
            if (i == arr[1].size()) return 0;
            if (j == arr[0].size()) return (long) 1e9;
            if (dp[i][j] != -1) return dp[i][j];
            long pick = Math.abs(arr[0].get(j) - arr[1].get(i)) + go(i + 1, j + 1);
            long leave = go(i, j + 1);
            dp[i][j] = Math.min(leave, pick);
            return dp[i][j];
        }

    }

    
}
