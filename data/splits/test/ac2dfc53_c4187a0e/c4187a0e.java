import java.io.OutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.io.PrintStream;
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
        CHiddenPermutations solver = new CHiddenPermutations();
        int testCount = Integer.parseInt(in.next());
        for (int i = 1; i <= testCount; i++)
            solver.solve(i, in, out);
        out.close();
    }

    static class CHiddenPermutations {
        public void solve(int testNumber, InputReader in, OutputWriter out) {
            int n = in.nextInt();
            int[] arr = new int[n];
            boolean[] vis = new boolean[n];
            for (int i = 0; i < n; i++) {
                if (vis[i]) continue;
                ArrayList<Integer> cycle = new ArrayList<>();
                int ans = query(i, in);
                int now = query(i, in);
                cycle.add(now);
                while (now != ans) {
                    now = query(i, in);
                    cycle.add(now);
                }
                for (int j = 0; j < cycle.size(); j++) {
                    arr[cycle.get(j)] = cycle.get((j + 1) % cycle.size()) + 1;
                    vis[cycle.get(j)] = true;
                }
            }
            out.print("! ");
            out.println(arr);
            out.flush();
        }

        int query(int ind, InputReader in) {
            System.out.println("? " + (ind + 1));
            return in.nextInt() - 1;
        }

    }

    

    
}
