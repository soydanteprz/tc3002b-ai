import java.io.OutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Stack;
import java.util.ArrayList;
import java.util.Vector;
import java.util.StringTokenizer;
import java.io.BufferedReader;
import java.io.InputStream;


public class Main {
    public static void main(String[] args) {
        InputStream inputStream = System.in;
        OutputStream outputStream = System.out;
        InputReader in = new InputReader(inputStream);
        PrintWriter out = new PrintWriter(outputStream);
        DDiscreteCentrifugalJumps solver = new DDiscreteCentrifugalJumps();
        solver.solve(1, in, out);
        out.close();
    }

    static class DDiscreteCentrifugalJumps {
        public void solve(int testNumber, InputReader in, PrintWriter out) {
            int n = in.nextInt();
            ArrayList<Integer> jumps[] = new ArrayList[n];
            int h[] = new int[n];
            for (int i = 0; i < n; i++) {
                h[i] = in.nextInt();
                jumps[i] = new ArrayList<>();
            }
            Stack<int[]> s = new Stack<>();
            for (int i = 0; i < n; i++) {
                while (s.size() > 0 && s.peek()[0] > h[i]) {
                    s.pop();
                }
                if (s.size() > 0) {
                    jumps[i].add(s.peek()[1]);
                }
                s.add(new int[]{h[i], i});
            }
            s.clear();
            for (int i = 0; i < n; i++) {
                while (s.size() > 0 && s.peek()[0] < h[i]) {
                    s.pop();
                }
                if (s.size() > 0) {
                    jumps[i].add(s.peek()[1]);
                }
                s.add(new int[]{h[i], i});
            }
            s.clear();
            for (int i = n - 1; i >= 0; i--) {
                while (s.size() > 0 && s.peek()[0] < h[i]) {
                    s.pop();
                }
                if (s.size() > 0) {
                    jumps[s.peek()[1]].add(i);
                }
                s.add(new int[]{h[i], i});
            }
            s.clear();
            for (int i = n - 1; i >= 0; i--) {
                while (s.size() > 0 && s.peek()[0] > h[i]) {
                    s.pop();
                }
                if (s.size() > 0) {
                    jumps[s.peek()[1]].add(i);
                }
                s.add(new int[]{h[i], i});
            }
            s.clear();
            int dp[] = new int[n];
            dp[0] = 0;
            for (int i = 1; i < n; i++) {
                dp[i] = Integer.MAX_VALUE;
                for (int x : jumps[i]) {
                    dp[i] = Math.min(dp[i], dp[x] + 1);
                }
            }
            out.print(dp[n - 1]);
        }

    }

    
}
