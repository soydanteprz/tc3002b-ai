import java.util.*;
import java.io.*;

public class C {
    static PrintWriter pw;

    void solve(int n, char[] a, char[] b) {
        int same = 0, diff = 0;
        int[] sameCnt = {0, 0}, diffCnt = {0, 0};
        for (int i = 0; i < n; i++) {
            if (a[i] == b[i]) {
                same++;
                sameCnt[a[i] - '0']++;
            } else {
                diff++;
                diffCnt[a[i] - '0']++;
            }
        }
        // tr(same, sameCnt, diff, diffCnt);
        int res = Integer.MAX_VALUE;
        if (same % 2 != 0 && sameCnt[1] - sameCnt[0] == 1) res = Math.min(res, same);
        if (diff % 2 == 0 && diffCnt[1] == diffCnt[0]) res = Math.min(res, diff);
        pr(res == Integer.MAX_VALUE ? -1 : res);
    }

    private void run() {
        // read_write_file(); // comment this before submission
        FastScanner fs = new FastScanner();
        int t = fs.nextInt();
        while (t-- > 0) {
            int n = fs.nextInt();
            char[] a = fs.next().toCharArray(), b = fs.next().toCharArray();
            solve(n, a, b);
        }
    }

    private final String INPUT = "input.txt";
    private final String OUTPUT = "output.txt";

    

    public static void main(String[] args) {
        pw = new PrintWriter(System.out);
        new C().run();
        pw.close();
    }

    void pr(int num) {
        pw.println(num);
    }

    void pr(long num) {
        pw.println(num);
    }

    void pr(double num) {
        pw.println(num);
    }

    void pr(String s) {
        pw.println(s);
    }

    void pr(char c) {
        pw.println(c);
    }

    

    
}
