import java.io.OutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.util.InputMismatchException;
import java.io.IOException;
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
        PrintWriter out = new PrintWriter(outputStream);
        CSbalansirovannieKuchkiKamnei solver = new CSbalansirovannieKuchkiKamnei();
        int testCount = Integer.parseInt(in.next());
        for (int i = 1; i <= testCount; i++)
            solver.solve(i, in, out);
        out.close();
    }

    static class CSbalansirovannieKuchkiKamnei {
        public void solve(int testNumber, InputReader in, PrintWriter out) {
            int n = in.nextInt();
            long[] a = new long[n];
            for (int i = 0; i < n; i++) {
                a[i] = in.nextInt();
            }
            long t1 = 0;
            long t2 = (int) 1e9 + 10;
            long[] b = new long[n];
            while (t1 < t2) {
                long mid = (t1 + t2 + 1) / 2;
                System.arraycopy(a, 0, b, 0, n);
                boolean fine = true;
                for (int i = n - 1; i >= 0; i--) {
                    if (b[i] < mid) {
                        fine = false;
                        break;
                    }
                    if (i >= 2) {
                        long how = Math.min(a[i], (b[i] - mid)) / 3;
                        b[i - 1] += how;
                        b[i - 2] += how * 2;
                    }
                }
                if (fine) {
                    t1 = mid;
                } else {
                    t2 = mid - 1;
                }
            }
            out.println(t1);
        }

    }

    
}
