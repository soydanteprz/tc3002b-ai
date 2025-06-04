import java.io.*;
import java.util.*;

public class Main {
    public static void main(String[] args) throws IOException {
        br = new BufferedReader(new InputStreamReader(System.in));
        PrintWriter pw = new PrintWriter(System.out);
        int n = nextInt();
        int k = nextInt();

        f = new int[n + 42];
        rf = new int[n + 42];
        f[0] = 1;
        rf[0] = 1;
        for (int i = 1; i < f.length; i++) {
            f[i] = mul(f[i - 1], i);
            rf[i] = mul(rf[i - 1], inv(i));
        }

        int[] a = new int[n * 2];
        for (int i = 0; i < n; i++) {
            a[i] = nextInt() * 2;
            a[i + n] = nextInt() * 2 + 1;
        }
        Arrays.sort(a);

        int ans = 0;
        int curOpen = 0;
        for (int r = 0; r < 2 * n;) {
            int l = r;
            while (r < 2 * n && a[l] == a[r]) r++;
            int intersections = r - l;
            if (a[l] % 2 == 0) {
                ans += C(curOpen + intersections, k);
                if (ans >= mod) ans -= mod;
                ans += mod - C(curOpen, k);
                if (ans >= mod) ans -= mod;
                curOpen += intersections;
            } else {
                curOpen -= intersections;
            }
        }

        pw.println(ans);
        pw.close();
    }

    static int mod = 998244353;

    static int mul(int a, int b) {
        return (int) ((long) a * (long) b % mod);
    }

    static int[] f;
    static int[] rf;

    static int C(int n, int k) {
        return (k < 0 || k > n) ? 0 : mul(f[n], mul(rf[n - k], rf[k]));
    }

    static int pow(int a, int n) {
        int res = 1;
        while (n != 0) {
            if ((n & 1) == 1) {
                res = mul(res, a);
            }
            a = mul(a, a);
            n >>= 1;
        }
        return res;
    }

    static int inv(int a) {
        return pow(a, mod - 2);
    }

    static StringTokenizer st = new StringTokenizer("");
    static BufferedReader br;

    static String next() throws IOException {
        while (st == null || !st.hasMoreTokens()) st = new StringTokenizer(br.readLine());
        return st.nextToken();
    }

    static int nextInt() throws IOException {
        return Integer.parseInt(next());
    }

    

    
}




