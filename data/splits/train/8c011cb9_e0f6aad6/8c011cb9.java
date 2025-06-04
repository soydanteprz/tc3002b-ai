import java.io.*;
import java.util.*;

/*
polyakoff
*/

public class Main {

    static FastReader in;
    static PrintWriter out;
    static Random rand = new Random();
    static final int oo = (int) 2e9 + 10;
    static final long OO = (long) 2e18 + 10;
    static final int MOD = 998244353;


    static void solve() {
        int n = in.nextInt();
        String[] s = new String[n];
        for (int i = 0; i < n; i++) {
            s[i] = in.next();
        }

        HashSet<String> set = new HashSet<>();
        for (int i = 0; i < n; i++) {
            if (s[i].charAt(0) == s[i].charAt(s[i].length() - 1)) {
                out.println("YES");
                return;
            }
            if (s[i].length() == 2) {
                String t = new StringBuilder(s[i]).reverse().toString();
                if (set.contains(t)) {
                    out.println("YES");
                    return;
                }
                for (char c = 'a'; c <= 'z'; c++) {
                    if (set.contains(t + c)) {
                        out.println("YES");
                        return;
                    }
                }
                set.add(s[i]);
            } else {
                String t = new StringBuilder(s[i]).reverse().toString();
                if (set.contains(t) || set.contains(t.substring(0, 2))) {
                    out.println("YES");
                    return;
                }
                set.add(s[i]);
            }
        }

        out.println("NO");


    }

    public static void main(String[] args) {
        in = new FastReader();
        out = new PrintWriter(System.out);

        int t = 1;
        t = in.nextInt();
        while (t-- > 0) {
            solve();
        }

        out.flush();
        out.close();
    }

    
}
