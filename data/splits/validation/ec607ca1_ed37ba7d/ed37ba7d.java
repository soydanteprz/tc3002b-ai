import java.io.*;
import java.util.*;
import java.lang.*;

public class codeforces {
    public static void main(String[] args) {
        InputStream inputStream = System.in;
        OutputStream outputStream = System.out;
        if (System.getProperty("ONLINE_JUDGE") == null) {
            long startTime = System.currentTimeMillis();
            try {
                sc = new InputReader(new FileInputStream("input.txt"));
                out = new PrintWriter(new FileOutputStream("output.txt"));
                pr = new PrintWriter(new FileOutputStream("error.txt"));
            } catch (Exception ignored) {
            }
            int t = 1;
            int tt = t;
            t = sc.nextInt();
            while (t-- > 0) {
                solve();
            }
            long endTime = System.currentTimeMillis();
            System.out.println("Time: " + (endTime - startTime) / tt + " ms");
            out.flush();
            pr.flush();

        } else {
            sc = new InputReader(inputStream);
            out = new PrintWriter(outputStream);
            pr = new PrintWriter(outputStream);
            int t = 1;
            t = sc.nextInt();
            while (t-- > 0) {
                solve();
            }
            out.flush();
        }
    }

    public static void solve() {
        n = sc.nextInt();
        for (int i = 0; i < n; i++) {
            dp[i] = inf;    
            ans[i] = inf;
        }
        m = sc.nextInt();
        for (int i = 0; i < m; i++)
            arr[i] = sc.nextInt() - 1;
        for (int i = 0; i < m; i++) {
            arr2[i] = sc.nextInt();
            dp[(int)arr[i]] = arr2[i];
        }
        temp = inf;
        for (int i = 0; i < n; i++) {
            temp = Math.min(temp, dp[i]);
            ans[i] = Math.min(ans[i], temp);
            temp++;
        }
        temp = inf;
        for (int i = (int)n - 1; i > -1; i--) {
            temp = Math.min(temp, dp[i]);
            ans[i] = Math.min(ans[i], temp);
            temp++;
        }
        for (int i = 0; i < n; i++)
            out.print(ans[i] + " ");
        out.println("");
    }
    /*
     * Set Iterator Iterator value = set.iterator(); Displaying the values after
     * iterating through the iterator
     * System.out.println("The iterator values are: "); while (value.hasNext()) {
     * System.out.println(value.next()); }
     */

    /*
     * Map Iterator: for (Map.Entry<Integer, Integer> entry : map.entrySet()){
     * System.out.println("Key => " + entry.getKey() + ", Value => " +
     * entry.getValue());}
     */
    // Globals

    public static long n, m, temp;
    public static int template_array_size = (int) 1e6 + 16813;
    public static long[] arr = new long[template_array_size];
    public static long[] arr2 = new long[template_array_size];
    public static long[] dp = new long[template_array_size];
    public static long[] ans = new long[template_array_size];
    public static int inf = Integer.MAX_VALUE;
    public static int minf = Integer.MIN_VALUE;
    public static int mod = 1000000007;
    public static int ml = (int) 1e9;
    public static String s = "";
    public static InputReader sc;
    public static PrintWriter out;
    public static PrintWriter pr;

    // Pair

    

    // FastReader Class

    

    // Req Functions
    

    

    

    

    

    

    

    

    

    

    

    

    

    

    

    

    

    

    

    

    

    

    

    

    // debug

    public static boolean sysFlag = System.getProperty("ONLINE_JUDGE") == null;

    

    

    

    

    

    

    

    

    

    

    

    

    

    

    

    

    

    

    

    
}
