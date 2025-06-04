import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.Random;
import java.util.StringTokenizer;

public final class C {

    public static void main(String[] args) {
        final FastScanner fs = new FastScanner();
        final int t = fs.nextInt();
        for (int test = 0; test < t; test++) {
            final int n = fs.nextInt();
            final int[] arr = fs.nextIntArray(n);
            long evenSum = arr[0];
            long oddSum = arr[1];
            int minEven = arr[0];
            int minOdd = arr[1];
            long res = (long) n * arr[0] + (long) n * arr[1];
            for (int i = 2; i < n; i++) {
                if (i % 2 == 0) {
                    minEven = Math.min(minEven, arr[i]);
                    evenSum += arr[i];
                    final long eC = n - i / 2 - 1;
                    final long oC = n - i / 2;
                    final long curr = evenSum + eC * minEven + oddSum + oC * minOdd;
                    res = Math.min(res, curr);

                } else {
                    minOdd = Math.min(minOdd, arr[i]);
                    oddSum += arr[i];
                    final long c = n - i / 2 - 1;
                    final long curr = oddSum + c * minOdd + evenSum + c * minEven;
                    res = Math.min(res, curr);
                }
            }
            System.out.println(res);
        }
    }

    static final class Utils {
        

        

        

        private Utils() {}
    }

    
}
