import java.util.*;

public class d {

    public static Scanner sc = new Scanner(System.in);
    public static final int MOD = 998244353;
    int []f; int [] lf;

    public static int mul(int a, int b) {
        return (int)((long)a * (long)b % MOD);
    }

    public static int ksm(int a, int n) {
        int ans = 1;
        while(n > 0) {
            if((n & 1) == 1)
                ans = mul(a, ans);
            a = mul(a, a);
            n >>= 1;
        }
        return ans;
    }

    public int C(int n, int k) {
        return (k < 0 || k > n) ? 0 : mul(f[n], mul(lf[n - k], lf[k]));
    }

    public static int inv(int a) {
        return ksm(a, MOD - 2);
    }

    public void solve() {
        int n = sc.nextInt();
        int k = sc.nextInt();
        f = new int[n + 42];
        lf = new int[n + 42];
        f[0] = lf[0] = 1;
        for(int i = 1; i < f.length; i++) {
            f[i] = mul(f[i  - 1], i);
            lf[i] = mul(lf[i - 1], inv(i));
        }

        int[] events = new int[2 * n];
        for(int i = 0; i < n; i++) {
            int le = sc.nextInt();
            int ri = sc.nextInt();
            events[i] = le * 2;
            events[i + n] = ri * 2 + 1;
        }

        Arrays.sort(events);
        int ans = 0, balance = 0;
        for(int r = 0; r < 2 * n;) {
            int l = r;
            while(r < 2 * n && events[l] == events[r]) ++r;
            int added = r - l;
            if(events[l] % 2 == 0) {
                ans += C(balance + added, k);
                if(ans >= MOD) ans -= MOD;
                ans += MOD - C(balance, k);
                if(ans >= MOD) ans -= MOD;
                balance += added;
            }
            else balance -= added;
        }
        sc.close();
        System.out.println(ans);
    }

    public static void main(String[] args) {
        (new d()).solve();
    }
}
