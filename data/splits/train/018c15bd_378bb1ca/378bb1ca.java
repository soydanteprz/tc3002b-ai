import java.util.*;

public class test {
    static final long mod = (long) 1e9 + 7;

    

    public static void main(String args[]) {

        Scanner sc = new Scanner(System.in);
        int t = 1;
        t = sc.nextInt();
        while (t-- > 0) {
            // Start code
            int n = sc.nextInt();
            long[] cost = new long[n];
            long[] a = new long[n];
            for (int i = 0; i < n; i++) {
                cost[i] = sc.nextLong();
                a[i] = cost[i];
                if (i > 1)
                    a[i] = Math.min(a[i], a[i - 2]);
            }
            long sum = cost[0];
            long res = Long.MAX_VALUE;

            for (int i = 1; i < n; i++) {
                sum += cost[i];
                long cur = sum - a[i] - a[i - 1];
                int p = (i + 1) / 2;
                int q = (i + 1) - p;
                int x = n - p + 1;
                int y = n - q + 1;
                cur += (a[i - 1] * x + a[i] * y);
                res = Math.min(res, cur);
            }
            println(res);
        }
        sc.close();
    }

    

    static void println(Object o) {
        System.out.println(o);
    }

    

    

    

    

    

    
}
