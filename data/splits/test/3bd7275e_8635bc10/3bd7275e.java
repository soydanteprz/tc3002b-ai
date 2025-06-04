import java.io.*;
import java.util.*;

public class tank {

    static final FastScanner fs = new FastScanner();

    static PrintWriter out = new PrintWriter(System.out);

    public static void main(String[] args) {
        int t = fs.nextInt();
        while(t-- > 0) run_case();

        out.close();
    }

    static void run_case() {
        int n = fs.nextInt();
        long k = fs.nextLong();
        int[] a = fs.readArray(n);

        Arrays.sort(a);

        long sum = 0;

        for(int i: a) sum += i;

        if(sum <= k) {
            out.println(0);
            return;
        }

        long ans = sum - k, dif;

        for (int i = 1; i < n; i++) {
            sum -= a[n - i];
            sum += a[0];
            //out.println(sum);
            if(sum <= k) {
                ans = Math.min(ans, i);
                continue;
            }
            dif = sum - k;
            if(dif % (i + 1) == 0) {
                ans = Math.min(ans, dif / (i + 1) + i);
            } else {
                ans = Math.min(ans, dif / (i + 1) + 1 + i);
            }
        }

        out.println(ans);
    }


    
}
