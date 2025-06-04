import java.lang.*;
import java.util.*;
import java.io.*;

public class Main {
    static FastScanner in = new FastScanner();

    static final int MOD = 998244353;
    static int n, k;
    static int[] fact, invFact;

    static int mul(int a, int b) {
        return (int)((long)a * (long)b % MOD);
    }

    static int pow(int a, int p) {
        int ans = 1;
        while (p != 0) {
            if ((p & 1) != 0)
                ans = mul(ans, a);
            a = mul(a, a);
            p >>= 1;
        }
        return ans;
    }

    static int inv(int a) {
        return pow(a, MOD - 2);
    }

    static void precalc() {
        fact = new int[n + 42]; invFact = new int[n + 42];
        fact[0] = invFact[0] = 1;
        for (int i = 1; i < fact.length; ++i) {
            fact[i] = mul(fact[i - 1], i);
            invFact[i] = mul(invFact[i - 1], inv(i));
        }
    }

    static int nCk(int a, int b) {
        if (b < 0 || a < b)
            return 0;
        return mul(fact[a], mul(invFact[a - b], invFact[b]));
    }

    static void solve() {
        n = in.nextInt(); k = in.nextInt();
        precalc();
        int[] events = new int[2 * n];
        for (int i = 0; i < n; ++i) {
            int l = in.nextInt(), r = in.nextInt();
            events[i] = 2 * l;
            events[i + n] = 2 * r + 1;
        }
        Arrays.sort(events);

        int ans = 0, balance = 0;
        for (int r = 0; r < 2 * n;) {
            int l = r;
            while (r < 2 * n && events[l] == events[r])
                ++r;
            int added = r - l;

            if (events[l] % 2 == 0) {
                ans += nCk(balance + added, k);
                if (ans >= MOD) ans -= MOD;
                ans += MOD - nCk(balance, k);
                if (ans >= MOD) ans -= MOD;
                balance += added;
            } else
                balance -= added;
        }
        System.out.println(ans);
    }

    public static void main(String[] args) {
        int T = 1;
        while (T-- > 0)
            solve();
    }

    
}


