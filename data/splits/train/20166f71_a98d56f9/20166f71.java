import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.io.PrintWriter;
import java.io.File;
import java.io.FileInputStream;
import java.util.*;

public class Main {

    // static final File ip = new File("input.txt");
    // static final File op = new File("output.txt");
    // static {
    // try {
    // System.setOut(new PrintStream(op));
    // System.setIn(new FileInputStream(ip));
    // } catch (Exception e) {
    // }
    // }
    private static final PrintWriter out = new PrintWriter(System.out);

    public static void main(String[] args) {
        FastReader sc = new FastReader();
        int test = 1;
        test = sc.nextInt();
        while (test-- > 0) {
            int n = sc.nextInt();
            long k = sc.nextLong();
            long[] a = new long[n];
            for (int i = 0; i < n; i++) {
                a[i] = sc.nextLong();
            }
            sortL(a);
            long[] pref = new long[n + 1];
            for (int i = 0; i < n; i++) {
                pref[i + 1] = pref[i] + a[i];
            }
            long ans = (long) 9e18;
            for (int t = 0; t <= n - 1; t++) {
                long sum = pref[n - t] + a[0] * t;
                long cur = t;
                if (sum > k) {
                    long diff = sum - k;
                    cur += (diff + t) / (t + 1);
                }
                ans = Math.min(ans, cur);
            }
            out.println(ans);
        }
        out.close();
    }

    

    

    

    

    

    

    

    

    

    

    

    

    

    

    

    

    

    

    static void sortL(long[] arr) {
        int n = arr.length;
        Random rnd = new Random();
        for (int i = 0; i < n; ++i) {
            long tmp = arr[i];
            int randomPos = i + rnd.nextInt(n - i);
            arr[i] = arr[randomPos];
            arr[randomPos] = tmp;
        }
        Arrays.sort(arr);
    }

    ////////////////////////////////// DSU START ///////////////////////////

    

    ////////////////////////////////// DSU END /////////////////////////////

    
}
