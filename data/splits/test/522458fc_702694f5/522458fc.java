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

    static int ask(int i) {
        out.println("? " + (i + 1));
        out.flush();
        return in.nextInt() - 1;
    }

    static void solve() {
        int n = in.nextInt();

        int[] p = new int[n];
        Arrays.fill(p, -1);
        for (int i = 0; i < n; i++) {
            if (p[i] == -1) {
                int first = ask(i);
                ArrayList<Integer> cycle = new ArrayList<>();
                while (true) {
                    int x = ask(i);
                    cycle.add(x);
                    if (x == first)
                        break;
                }
                int sz = cycle.size();
                int from = -1;
                for (int j = 0; j < sz; j++) {
                    if (cycle.get(j) == i) {
                        from = j;
                        break;
                    }
                }
                p[i] = cycle.get((from + 1) % sz);
                for (int j = (from + 1) % sz; j != from; j = (j + 1) % sz) {
                    p[cycle.get(j)] = cycle.get((j + 1) % sz);
                }
            }
        }

        out.print("!");
        for (int i = 0; i < n; i++) {
            out.print(" " + (p[i] + 1));
        }
        out.println();
        out.flush();

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
