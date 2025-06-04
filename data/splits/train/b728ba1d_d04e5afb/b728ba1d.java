import java.io.OutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.util.InputMismatchException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Objects;
import java.util.Collections;
import java.io.InputStream;

public class Main {
    public static void main(String[] args) {
        InputStream inputStream = System.in;
        OutputStream outputStream = System.out;
        FastReader in = new FastReader(inputStream);
        PrintWriter out = new PrintWriter(outputStream);
        CPhoenixAndTowers solver = new CPhoenixAndTowers();
        int testCount = Integer.parseInt(in.next());
        for (int i = 1; i <= testCount; i++)
            solver.solve(i, in, out);
        out.close();
    }

    static class CPhoenixAndTowers {
        public void solve(int testNumber, FastReader in, PrintWriter out) {
            int n = in.nextInt(), m = in.nextInt(), k = in.nextInt();
            ArrayList<Pair<Integer, Integer>> a = new ArrayList<>();
            for (int i = 0; i < n; ++i) {
                a.add(new Pair<>(in.nextInt(), i));
            }
            Collections.sort(a);

            int[] ans = new int[n];
            int[] sum = new int[m];
            int j = 1;

            for (int i = 0; i < n; ++i) {
                ans[a.get(i).y] = j;
                sum[j - 1] += a.get(i).x;
                j++;
                if (j == m + 1) j = 1;
            }

            for (int i = 1; i < m; ++i) {
                if (Math.abs(sum[i - 1] - sum[i]) > k) {
                    out.println("NO");
                }
            }

            out.println("YES");
            for (int e : ans) {
                out.print(e + " ");
            }
            out.println();

        }

    }

    static class Pair<U, V> implements Comparable<Pair<U, V>> {
        public U x;
        public V y;

        public Pair(U x, V y) {
            this.x = x;
            this.y = y;
        }

        

        

        

    }

    
}
