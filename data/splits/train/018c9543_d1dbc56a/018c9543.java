import java.io.*;
import java.util.*;

public class Main {
    public static int n;
    public static int a[] = new int[12];
    public static int b[] = new int[12];

    public static boolean f(int p) {
        if (p > n) return false;

        for (int i = 0; i < p; i++) {
            b[p] = a[p] + b[i];
            for (int j = 0; j < p; j++) {
                if (b[j] == b[p])
                    return true;
            }
            if (f(p + 1)) {
                return true;
            }


//            b[p] = b[i] - a[p];
//            for (int j = 0; j < p; j++) {
//                if (b[j] == b[p])
//                    return true;
//            }
//            if (f(p + 1)) {
//                return true;
//            }
        }
        return false;
    }

    public static void main(String[] args) throws Exception {
        int T = r.readInt();
        for (int t = 0; t < T; t++) {
            n = r.readInt();
            for (int i = 1; i <= n; i++) {
                a[i] = r.readInt();
            }

            boolean ans = false;
            if (n == 1) {
                ans = a[1] == 0;
            } else {
                b[0] = 1;
                ans = f(1);
            }

            if (ans) {
                System.out.println("YES");
            } else {
                System.out.println("NO");
            }
        }
    }

    static public InputReader r = new InputReader(System.in);
    static public OutputWriter w = new OutputWriter(System.out);

    

    
}
