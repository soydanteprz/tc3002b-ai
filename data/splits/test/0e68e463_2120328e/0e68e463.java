import java.io.*;
import java.util.*;

import static java.lang.Math.*;

public class D {
    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        PrintWriter pw = new PrintWriter(System.out);
        int t = in.nextInt();
        for(int tt = 0; tt < t; tt++) {
            int n = in.nextInt();
            long[] arr = new long[n];
            for (int i = 0; i < n; i++) arr[i] = in.nextLong();
            char[] s = in.next().toCharArray();
            if (solve(arr, n, s))pw.println("YES");
            else pw.println("NO");
        }
        pw.close();
    }
    static boolean solve(long[] arr, int n, char[] s) {
        ArrayList<Long> B = new ArrayList<>();
        ArrayList<Long> R = new ArrayList<>();
        for(int i = 0; i < n; i++) {
            if (s[i] == 'B') B.add(arr[i]);
            else R.add(arr[i]);
        }
        Collections.sort(B);
        Collections.sort(R);
      //  debug(B);
        long last = n;
        for (int i = R.size() - 1; i >= 0; i--) {
            long v = R.get(i);
            if (v > last) {
               return false;
            }
            last--;
        }
        long first = 1;
        int size = B.size();
        for (int i = 0; i < size; i++) {
            long v = B.get(i);
         //   debug(v, first);
            if (v < first) return false;
            first++;
        }
        return true;
    }


    
}
