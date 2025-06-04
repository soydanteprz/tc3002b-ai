import java.io.OutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.util.Arrays;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.StringTokenizer;
import java.io.Writer;
import java.io.OutputStreamWriter;
import java.io.BufferedReader;
import java.io.InputStream;

/**
 * Built using CHelper plug-in
 * Actual solution is at the top
 */
public class Main {
    public static void main(String[] args) {
        InputStream inputStream = System.in;
        OutputStream outputStream = System.out;
        InputReader in = new InputReader(inputStream);
        OutputWriter out = new OutputWriter(outputStream);
        DArmchairs solver = new DArmchairs();
        solver.solve(1, in, out);
        out.close();
    }

    static class DArmchairs {
        static ArrayList<Integer> empty;
        static ArrayList<Integer> chair;
        static long[][] dp;

        public void solve(int testNumber, InputReader in, OutputWriter out) {
            int n = in.nextInt();
            int[] arr = in.nextIntArray(n);
            empty = new ArrayList<>();
            chair = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                int a = arr[i];
                if (a == 1) chair.add(i);
                else empty.add(i);
            }
            dp = new long[chair.size()][empty.size()];
            for (long[] row : dp) Arrays.fill(row, -1);
            long ans = rec(0, 0);
            out.println(ans);
        }

        static long rec(int i, int j) {
            if (i == chair.size()) return 0;
            int req = chair.size() - i;
            int have = empty.size() - j;
            if (req > have) return Integer.MAX_VALUE;

            if (dp[i][j] != -1) return dp[i][j];
            long opt1 = Math.abs(chair.get(i) - empty.get(j)) + rec(i + 1, j + 1);
            long opt2 = rec(i, j + 1);
            dp[i][j] = Math.min(opt1, opt2);
            return dp[i][j];
        }

    }

    

    
}
