import java.io.DataInputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;

public class Main {

    private static void run() throws IOException {
        int n, k;
        n = in.nextInt();
        k = in.nextInt();
        int[] a = new int[n];
        long[] t = new long[k];

        for (int i = 0; i < n; i++) {
            a[i] = -1;
        }
        for (int i = 0; i < k; i++) {
            a[in.nextInt() - 1] = i;
        }
        for (int i = 0; i < k; i++) {
            t[i] = in.nextLong();
        }

        long[] dp = new long[n];

        long now;
        now = 2000000000;
        for (int i = 0; i < n; i++) {
            now++;
            if (a[i] != -1) {
                now = Math.min(now, t[a[i]]);
            }

            dp[i] = now;
        }
        now = 2000000000;
        for (int i = n - 1; i >= 0; i--) {
            now++;
            if (a[i] != -1) {
                now = Math.min(now, t[a[i]]);
            }

            dp[i] = Math.min(dp[i], now);
        }
        for (int i = 0; i < n; i++) {
            out.printf("%d ", dp[i]);
        }
        out.println();
    }

    public static void main(String[] args) throws IOException {
        in = new Reader();
        out = new PrintWriter(new OutputStreamWriter(System.out));

        int t = in.nextInt();
        for (int i = 0; i < t; i++) {
            run();
        }

        out.flush();
        in.close();
        out.close();
    }

    

    static final long mod = 1000000007;

    

    

    @SuppressWarnings("FieldCanBeLocal")
    private static Reader in;
    private static PrintWriter out;

    

    

    

    

    
}
