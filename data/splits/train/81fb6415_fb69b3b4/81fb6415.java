import java.math.BigInteger;
import java.util.*;
import java.io.*;
import java.util.concurrent.atomic.AtomicIntegerFieldUpdater;

public class CodeForces {


    public void run() throws Exception {
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        int t = Integer.parseInt(br.readLine());
        next : while (t-- > 0) {
            StringTokenizer st = new StringTokenizer(br.readLine());
            int n = Integer.parseInt(st.nextToken());
            long k = Long.parseLong(st.nextToken());
            Long[] a = new Long[n];
            st = new StringTokenizer(br.readLine());
            for (int i = 0; i < n; i++) {
                a[i] = Long.parseLong(st.nextToken());
            }
            Arrays.sort(a);
            long ans = Long.MAX_VALUE;
            long[] lsum = new long[n + 1];
            for (int i = 0; i < n; i++) {
                lsum[i + 1] = lsum[i] + a[i];
            }
            for (long y = 0; y < n; y++) {
                long x = 0;
                if ((k - lsum[n - (int)y] + a[0]) >= 0) {
                    x = (k - lsum[n - (int)y] + a[0]) / (y + 1);
                } else {
                    if ((k - lsum[n - (int)y] + a[0]) % (y + 1) == 0) {
                        x = (k - lsum[n - (int)y] + a[0]) / (y + 1);
                    } else {
                        x = (k - lsum[n - (int)y] + a[0]) / (y + 1) - 1;
                    }
                }
                x = a[0] - x;
                ans = Math.min(ans, Math.max(0, x) + y);
            }
            System.out.println(ans);
        }

    }

    public static void main(String[] args) throws Exception {
        new CodeForces().run();
    }

}
