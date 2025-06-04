import java.util.*;
import java.io.*;

public class _724 {
    public static void main(String[] args) {
        MyScanner sc = new MyScanner();
        PrintWriter out = new PrintWriter(new BufferedOutputStream(System.out));
        int t = sc.nextInt();
        while (t-- > 0) {
            int n = sc.nextInt();
            TreeSet<Long> set = new TreeSet<>();
            long prev = -1;
            boolean ok = true;
            for (int i = 0; i < n; i++) {
                long x = sc.nextInt();
                if (i == 0) {
                    set.add(x);
                    prev = x;
                    continue;
                }
                if (x > prev) {
                    Long high = set.higher(prev);
                    if (high == null) set.add(x);
                    else if (high >= x) set.add(x);
                    else {
                        ok = false;
                    }
                } else if (x < prev) {
                    Long low = set.lower(prev);
                    if (low == null) set.add(x);
                    else if (low <= x) set.add(x);
                    else {
                        ok = false;
                    }
                }
                prev = x;
            }
            out.println(ok ? "YES" : "NO");
        }
        out.close();
    }





    

    


    //-----------MyScanner class for faster input----------
    

}
