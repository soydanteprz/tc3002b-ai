import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.math.BigInteger;
import java.text.DecimalFormat;
import java.util.*;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.LongStream;

public class TaskB {

    static long mod = 1000000007;
    static FastScanner scanner;

    static final StringBuilder result = new StringBuilder();

    public static void main(String[] args) {
        // 2 : 1000000000
        scanner = new FastScanner();
        int T = scanner.nextInt();
        for (int t = 0; t < T; t++) {
            solve(t + 1);
            result.append("\n");
        }
        System.out.println(result);
    }

    static void solve(int t) {
        int n = scanner.nextInt();
        int[] a = scanner.nextIntArray(n);
        String s = scanner.nextToken();

        List<Integer> blue = new ArrayList<>();
        List<Integer> red = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            if (s.charAt(i) == 'B') {
                blue.add(a[i]);
            } else {
                red.add(a[i]);
            }
        }
        Collections.sort(blue);
        Collections.sort(red);
        for (int i = 0; i < blue.size(); i++) {
            if (blue.get(i) < i + 1) {
                result.append("NO");
                return;
            }
        }
        for (int i = 0; i < red.size(); i++) {
            if (red.get(i) > i + 1 + blue.size()) {
                result.append("NO");
                return;
            }
        }
        result.append("YES");
    }

    

    

    

    

    
}
