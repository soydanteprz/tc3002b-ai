import java.io.*;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;
import java.util.StringTokenizer;

public class codeforces_750_D {
    private static void solve(FastReader io) {
        int n = io.nextInt();
        int[] a = io.readArray(n);

        if (n % 2 == 0) {
            for (int i = 0; i < n; i += 2) {
                io.out.print(a[i + 1] + " ");
                io.out.print(-a[i] + " ");
            }
        } else {
            for (int i = 0; i < n - 3; i += 2) {
                io.out.print(a[i + 1] + " ");
                io.out.print(-a[i] + " ");
            }
            if (a[n - 3] + a[n - 2] != 0) {
                io.out.print(-a[n - 1] + " ");
                io.out.print(-a[n - 1] + " ");
                io.out.print(a[n - 3] + a[n - 2] + " ");
            } else if (a[n - 3] + a[n - 1] != 0) {
                io.out.print(-a[n - 2] + " ");
                io.out.print(a[n - 3] + a[n - 1] + " ");
                io.out.print(-a[n - 2] + " ");
            } else {
                io.out.print(a[n - 2] + a[n - 1] + " ");
                io.out.print(-a[n - 3] + " ");
                io.out.print(-a[n - 3] + " ");
            }
        }
        io.out.println();
    }

    public static void main(String[] args) throws Exception {
        try (FastReader ioAdapter = new FastReader()) {
            int count = 1;
            count = ioAdapter.nextInt();
            while (count-- > 0) {
                solve(ioAdapter);
            }
        }
    }

    
}
