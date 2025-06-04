import java.io.OutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.InputMismatchException;
import java.io.IOException;
import java.io.InputStream;

public class Main {
    public static void main(String[] args) {
        InputStream inputStream = System.in;
        OutputStream outputStream = System.out;
        FastReader in = new FastReader(inputStream);
        PrintWriter out = new PrintWriter(outputStream);
        EAirConditioners solver = new EAirConditioners();
        int testCount = Integer.parseInt(in.next());
        for (int i = 1; i <= testCount; i++)
            solver.solve(i, in, out);
        out.close();
    }

    static class EAirConditioners {
        public void solve(int testNumber, FastReader in, PrintWriter out) {
            int n = in.nextInt(), q = in.nextInt();
            int[] a = in.readArray(q);
            long[] t = in.readLongArray(q);
            long[] pref = new long[n + 2];
            long[] suff = new long[n + 2];

            Arrays.fill(pref, (long) 1e14);
            Arrays.fill(suff, (long) 1e14);

            for (int i = 0; i < q; ++i) {
                pref[a[i]] = t[i] - a[i];
                suff[a[i]] = t[i] + a[i];
            }

            for (int i = 1; i <= n; ++i) {
                pref[i] = Math.min(pref[i], pref[i - 1]);
            }

            for (int i = n; i >= 1; --i) {
                suff[i] = Math.min(suff[i], suff[i + 1]);
            }

            for (int i = 1; i <= n; ++i) {
                out.print(Math.min(pref[i] + i, suff[i] - i) + " ");
            }
            out.println();
        }

    }

    
}
