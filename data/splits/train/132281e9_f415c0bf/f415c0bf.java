import java.util.Scanner;

public class BalancedStoneHeaps {

    static boolean check(long[] a, long k) {
        int n = a.length;
        long[] temp = new long[n];
        for (int i = 0; i < n; i++) {
            temp[i] = a[i];
        }
        for (int i = n - 1; i >= 2; i--) {
            if (temp[i] < k) {
                
                return false;
            }
            long p = Math.min(a[i] / 3, (temp[i] - k) / 3);
            temp[i - 1] += p;
            temp[i - 2] += 2 * p;

        }

        if (temp[0] >= k && temp[1] >= k) {
            return true;
        }

        return false;

    }

    public static void main(String[] args) {
        Scanner scan = new Scanner(System.in);
        int t = scan.nextInt();
        while (t-- > 0) {
            int n = scan.nextInt();
            long[] a = new long[n];
            for (int i = 0; i < n; i++) {
                a[i] = scan.nextInt();
            }
            long l = 0;
            long r = a[n - 1];
            long res = 0;
            while (l <= r) {
                long mid = (l + r) / 2;
                if (check(a, mid)) {
                    l = mid + 1;
                    res = Math.max(res, mid);
                } else {
                    r = mid - 1;
                }
            }
            System.out.println(res);
        }
    }
}
